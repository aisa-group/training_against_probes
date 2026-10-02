
import os
import gc
import multiprocessing as mp

import torch

_TOKENIZER_PATCHED = False


def _hf_peft_generate(
    model_name: str,
    lora_path: str,
    tokenizer_name,
    formatted_prompts: list,
    max_new_tokens: int,
) -> list[str]:
    """HuggingFace PEFT fallback for architectures vLLM can't serve with LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok_id = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    completions = []
    batch_size = 8
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, ids in enumerate(out):
            prompt_len = enc["input_ids"].shape[1]
            completions.append(tokenizer.decode(ids[prompt_len:], skip_special_tokens=True))
    return completions


def _vllm_subprocess_worker(
    formatted_prompts: list,
    model_name: str,
    tokenizer_name,
    lora_path,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    lora_rank: int,
    result_queue: mp.Queue,
    tp_size: int = 0,
) -> None:
    """Runs vLLM entirely inside a subprocess so its GPU memory is freed on exit.

    ``tp_size``: tensor-parallel size. 0 (default) auto-detects via
    ``torch.cuda.device_count()`` so multi-GPU jobs (e.g. 70B on 4×A100)
    shard the model automatically. Pass an explicit int to override.
    """
    try:
        import os
        os.environ.setdefault("VLLM_RPC_GET_DATA_TIMEOUT_MS", "120000")  # 2 min (default is 5s)
        _patch_vllm_tokenizer_compat()
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        if tp_size <= 0:
            import torch
            tp_size = max(1, torch.cuda.device_count())

        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        use_lora = lora_path is not None

        try:
            llm = make_vllm_llm(
                model_id=model_name,
                tokenizer_id=tokenizer_name,
                tp_size=tp_size,
                gpu_mem_util=gpu_memory_utilization,
                dtype="float16",
                enable_lora=use_lora,
                max_lora_rank=lora_rank,
            )
            lora_request = LoRARequest("finetune_adapter", 1, lora_path) if use_lora else None
            outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
            completions = [out.outputs[0].text for out in outputs]
        except AssertionError as exc:
            if use_lora and "TransformersModel does not support LoRA" in str(exc):
                # vLLM fell back to TransformersModel (no native impl for this architecture).
                # Generate directly with HF PEFT instead.
                print(f"[vllm] LoRA unsupported by TransformersModel, falling back to HF PEFT generate")
                completions = _hf_peft_generate(
                    model_name, lora_path, tokenizer_name, formatted_prompts, max_new_tokens
                )
            else:
                raise

        result_queue.put(completions)
    except Exception as exc:
        result_queue.put(exc)


def _patch_vllm_tokenizer_compat():
    """
    Monkey-patch vLLM's get_cached_tokenizer to handle the removal of
    ``all_special_tokens_extended`` in newer transformers versions.

    vLLM 0.7.3 unconditionally accesses this attribute, which no longer exists
    on ``TokenizersBackend`` in transformers >=4.50.  We add it as an instance
    attribute (falling back to ``all_special_tokens``) before the original
    function runs, so ``__getattribute__`` finds it directly and never falls
    through to ``__getattr__``.
    """
    global _TOKENIZER_PATCHED
    if _TOKENIZER_PATCHED:
        return
    try:
        import vllm.transformers_utils.tokenizer as _vt
        _orig = _vt.get_cached_tokenizer

        def _patched(tokenizer):
            if not hasattr(tokenizer, "all_special_tokens_extended"):
                try:
                    tokenizer.all_special_tokens_extended = list(
                        getattr(tokenizer, "all_special_tokens", [])
                    )
                except Exception:
                    pass
            return _orig(tokenizer)

        _vt.get_cached_tokenizer = _patched
        _TOKENIZER_PATCHED = True
    except Exception:
        pass


def make_vllm_llm(
    model_id: str,
    tokenizer_id,
    tp_size: int,
    gpu_mem_util: float,
    dtype: str,
    enable_lora: bool = False,
    max_lora_rank: int = 64,
    max_model_len: int | None = None,
):
    import os as _os
    _patch_vllm_tokenizer_compat()
    from vllm import LLM
    kwargs = dict(
        model=model_id,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem_util,
        dtype=dtype,
        enforce_eager=False,
        enable_lora=enable_lora,
    )
    if enable_lora:
        kwargs["max_lora_rank"] = max_lora_rank
    # Multi-GPU custom all-reduce hangs/errors on some nodes during engine
    # warmup (e.g. "custom_all_reduce.cuh invalid argument"). Pass the LLM kwarg
    # explicitly when requested via env, since vLLM 0.7.3 may not read the env.
    if _os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE") == "1":
        kwargs["disable_custom_all_reduce"] = True
    if tokenizer_id is not None:
        kwargs["tokenizer"] = tokenizer_id
    # Cap context to fit KV cache budget — Llama-3.1's native 131072-token context
    # blows the KV cache when only ~50% of GPU mem is available for it.
    if max_model_len is None:
        max_model_len = int(_os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
    kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def destroy_vllm_llm(llm) -> None:
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _wildguard_subprocess_worker(
    token_ids_list: list,
    model_name: str,
    gpu_memory_utilization: float,
    result_queue: mp.Queue,
) -> None:
    """Runs WildGuard vLLM entirely inside a subprocess so its GPU memory is freed on exit."""
    try:
        import os
        os.environ.setdefault("VLLM_RPC_GET_DATA_TIMEOUT_MS", "120000")
        _patch_vllm_tokenizer_compat()
        from vllm import SamplingParams

        llm = make_vllm_llm(
            model_id=model_name,
            tokenizer_id=None,
            tp_size=1,
            gpu_mem_util=gpu_memory_utilization,
            dtype="float16",
        )
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in token_ids_list],
            SamplingParams(temperature=0, max_tokens=32),
        )
        texts = [o.outputs[0].text for o in outputs]
        result_queue.put(texts)
    except Exception as exc:
        result_queue.put(exc)


def run_wildguard_subprocess(
    token_ids_list: list,
    model_name: str,
    gpu_memory_utilization: float = 0.5,
) -> list[str]:
    """
    Run WildGuard vLLM inference in a subprocess so all GPU memory is released on exit.

    Using destroy_vllm_llm() in-process corrupts the CUDA context via destroy_model_parallel(),
    causing subsequent vLLM inits to report no GPU available.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_wildguard_subprocess_worker,
        args=(token_ids_list, model_name, gpu_memory_utilization, result_queue),
    )
    proc.start()
    result = result_queue.get()
    proc.join()
    if isinstance(result, Exception):
        raise result
    return result


def generate_vllm_subprocess(
    formatted_prompts: list,
    model_name: str,
    tokenizer_name,
    lora_path,
    max_new_tokens: int = 200,
    gpu_memory_utilization: float = 0.85,
    lora_rank: int = 64,
    timeout: int = 3600,
) -> list[str]:
    """
    Run vLLM generation in a subprocess so all GPU memory is released on exit.

    This is more reliable than destroy_vllm_llm() for freeing KV-cache memory,
    because the OS reclaims all CUDA allocations when the child process exits.

    The load timeout can be raised via the VLLM_SUBPROC_TIMEOUT env var (seconds)
    — large sharded models (e.g. 70B) on a busy shared filesystem can take longer
    than the 1-hour default to load into vLLM.
    """
    import os as _os
    timeout = int(_os.environ.get("VLLM_SUBPROC_TIMEOUT", timeout))
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_vllm_subprocess_worker,
        args=(
            formatted_prompts,
            model_name,
            tokenizer_name,
            lora_path,
            max_new_tokens,
            gpu_memory_utilization,
            lora_rank,
            result_queue,
        ),
    )
    proc.start()
    try:
        result = result_queue.get(timeout=timeout)
    except Exception:
        proc.kill()
        proc.join()
        raise RuntimeError(
            f"vLLM subprocess timed out after {timeout}s loading {model_name}"
        )
    proc.join()
    if isinstance(result, Exception):
        raise result
    return result


def _vllm_subprocess_worker_per_prompt(
    formatted_prompts: list,
    per_prompt_max_tokens: list,
    model_name: str,
    tokenizer_name,
    lora_path,
    gpu_memory_utilization: float,
    lora_rank: int,
    result_queue: mp.Queue,
    tp_size: int = 0,
) -> None:
    """vLLM worker that supports per-prompt max_tokens via parallel SamplingParams.

    ``tp_size``: tensor-parallel size; 0 (default) auto-detects via
    ``torch.cuda.device_count()`` so 70B-scale models shard across all visible GPUs.
    """
    try:
        import os
        os.environ.setdefault("VLLM_RPC_GET_DATA_TIMEOUT_MS", "120000")
        _patch_vllm_tokenizer_compat()
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        if tp_size <= 0:
            import torch
            tp_size = max(1, torch.cuda.device_count())

        sampling_params = [
            SamplingParams(temperature=0, max_tokens=int(mt))
            for mt in per_prompt_max_tokens
        ]
        use_lora = lora_path is not None

        try:
            llm = make_vllm_llm(
                model_id=model_name,
                tokenizer_id=tokenizer_name,
                tp_size=tp_size,
                gpu_mem_util=gpu_memory_utilization,
                dtype="float16",
                enable_lora=use_lora,
                max_lora_rank=lora_rank,
            )
            lora_request = LoRARequest("finetune_adapter", 1, lora_path) if use_lora else None
            outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
            completions = [out.outputs[0].text for out in outputs]
        except AssertionError as exc:
            if use_lora and "TransformersModel does not support LoRA" in str(exc):
                # vLLM fell back to TransformersModel (no native impl) — go through HF PEFT.
                # Use the largest per-prompt budget; HF generate stops at EOS.
                print(f"[vllm] LoRA unsupported by TransformersModel, falling back to HF PEFT generate")
                completions = _hf_peft_generate(
                    model_name, lora_path, tokenizer_name,
                    formatted_prompts, max(int(mt) for mt in per_prompt_max_tokens),
                )
            else:
                raise

        result_queue.put(completions)
    except Exception as exc:
        result_queue.put(exc)


def generate_vllm_subprocess_per_prompt(
    formatted_prompts: list,
    per_prompt_max_tokens: list,
    model_name: str,
    tokenizer_name,
    lora_path,
    gpu_memory_utilization: float = 0.85,
    lora_rank: int = 64,
    timeout: int = 3600,
) -> list[str]:
    """
    Like generate_vllm_subprocess but with per-prompt max_tokens.

    Lets multiple eval workloads (e.g. MMLU's 8-token answers and GSM8K's
    512-token CoT) share one vLLM subprocess instead of paying its ~30-60s
    cold start three times.
    """
    if len(formatted_prompts) != len(per_prompt_max_tokens):
        raise ValueError(
            f"len(prompts)={len(formatted_prompts)} != "
            f"len(per_prompt_max_tokens)={len(per_prompt_max_tokens)}"
        )
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_vllm_subprocess_worker_per_prompt,
        args=(
            formatted_prompts,
            per_prompt_max_tokens,
            model_name,
            tokenizer_name,
            lora_path,
            gpu_memory_utilization,
            lora_rank,
            result_queue,
        ),
    )
    proc.start()
    try:
        result = result_queue.get(timeout=timeout)
    except Exception:
        proc.kill()
        proc.join()
        raise RuntimeError(
            f"vLLM subprocess timed out after {timeout}s loading {model_name}"
        )
    proc.join()
    if isinstance(result, Exception):
        raise result
    return result
