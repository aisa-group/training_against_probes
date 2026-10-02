"""Evaluate MMLU, GSM8K, and IFEval through lm-evaluation-harness.

Shared by training and standalone checkpoint evaluation. Uses zero-shot tasks
and returns utility-prefixed metric keys. MMLU_LIMIT applies per subject;
GSM8K_LIMIT and IFEVAL_LIMIT apply to their entire task.

Evaluation cache settings are scoped to worker processes to preserve the
training process's dataset cache configuration.
"""
import os

# Optional external lm-evaluation-harness package directory (lm-eval is
# normally installed as a dependency, so this is only used when set).
LMEVAL_PKGS = os.environ.get("LMEVAL_PKGS", "")
if LMEVAL_PKGS and os.path.isdir(LMEVAL_PKGS):
    import sys as _sys
    if LMEVAL_PKGS not in _sys.path:
        _sys.path.insert(0, LMEVAL_PKGS)

# Environment for the evaluation worker process only, so training keeps its own
# settings. Cache overrides apply only when the LMEVAL_* variables are set.
# vLLM >= 0.8 defaults to its V1 engine, whose LoRA layers fail on the prompt
# log-probabilities lm-eval requests; the worker therefore uses the V0 engine.
EVAL_ENV = {
    k: v for k, v in {
        "HF_DATASETS_CACHE": os.environ.get("LMEVAL_DATASETS_CACHE"),
        "HF_HOME": os.environ.get("LMEVAL_HF_HOME"),
        "VLLM_USE_V1": os.environ.get("LMEVAL_VLLM_USE_V1", "0"),
    }.items() if v
}

# Tasks with likelihood-based multiple-choice scoring.
MULTIPLE_CHOICE_TASKS = {"mmlu"}
# Recognize generative MMLU results when harvesting metrics.
REASONING_MMLU_TASK = "mmlu_generative"
# Allow reasoning generation to continue past newlines; strip the trace before scoring.
GEN_UNTIL_EOS = "until=</s>"

THINK_END_TOKEN = "</think>"
# Configurable generation budget for reasoning models.
REASONING_GEN_TOKS = int(os.environ.get("REASONING_GEN_TOKS", "4096"))

GSM8K_LIMIT = 100
IFEVAL_LIMIT = 100
MMLU_LIMIT = 5              # per subject; 57 subjects -> 285 questions

# Metric keys shared by training and checkpoint evaluation.
K_MMLU = "utility/mmlu_accuracy"
K_GSM8K = "utility/gsm8k_accuracy"
K_IFEVAL_P = "utility/ifeval_prompt_accuracy"
K_IFEVAL_I = "utility/ifeval_instruction_accuracy"


def tasks(mmlu_limit=MMLU_LIMIT, gsm8k_limit=GSM8K_LIMIT, ifeval_limit=IFEVAL_LIMIT,
          reasoning=False):
    """reasoning is accepted and unused: the task list is the same either way.
    What changes for a reasoning model is the chat template, in evaluate()."""
    return [("gsm8k_cot_zeroshot", gsm8k_limit),   # zero-shot variant
            ("mmlu", mmlu_limit),                  # limit is PER subject
            ("ifeval", ifeval_limit)]


def harvest(results, out=None):
    """Extract utility metrics from lm-evaluation-harness results.

    GSM8K uses flexible answer extraction and also retains the strict metric.
    MMLU uses the harness's size-weighted aggregate across subjects.
    Result keys include the filter name, such as exact_match,flexible-extract.
    """
    r = results.get("results", {})
    out = {} if out is None else out

    def pick(task, name):
        for k, v in r.get(task, {}).items():
            if (k == name or k.startswith(name + ",")) and isinstance(v, (int, float)):
                return float(v)
        return None

    def gsm(kind):
        for k, v in r.get("gsm8k_cot_zeroshot", {}).items():
            if k.startswith("exact_match") and kind in k:
                return float(v)
        return None

    found = {
        K_GSM8K: gsm("flexible"),
        "utility/gsm8k_accuracy_strict": gsm("strict"),
        # the loglikelihood variant reports acc; mmlu_generative reports
        # exact_match. Take whichever ran.
        K_MMLU: (pick("mmlu", "acc")
                 if pick("mmlu", "acc") is not None
                 else pick(REASONING_MMLU_TASK, "exact_match")),
        K_IFEVAL_P: pick("ifeval", "prompt_level_strict_acc"),
        K_IFEVAL_I: pick("ifeval", "inst_level_strict_acc"),
        "utility/ifeval_prompt_accuracy_loose": pick("ifeval", "prompt_level_loose_acc"),
        "utility/ifeval_instruction_accuracy_loose": pick("ifeval", "inst_level_loose_acc"),
    }
    # Require a metric for each evaluated task to avoid counting missing results as zero.
    ran = {"gsm8k_cot_zeroshot": [K_GSM8K], "mmlu": [K_MMLU],
           REASONING_MMLU_TASK: [K_MMLU],
           "ifeval": [K_IFEVAL_P, K_IFEVAL_I]}
    missing = [k for t, ks in ran.items() if t in r for k in ks if found[k] is None]
    if missing:
        raise ValueError(f"lm_eval returned no value for {missing}; "
                         f"available: { {t: list(d) for t, d in r.items()} }")
    out.update({k: v for k, v in found.items() if v is not None})
    return out


def vllm_supports(model_name):
    """Can vLLM 0.7.3 serve this architecture?

    The container pins vLLM 0.7.3 for its tokenizer patch and torch 2.5.1, and
    that release predates Qwen3: Qwen3ForCausalLM first appears in vLLM 0.8.4,
    which requires torch 2.6. So Qwen3 models have to be scored through the HF
    backend instead. Detected from the model's own config rather than a config
    flag, so a new architecture needs no edit here to be handled correctly.
    """
    from transformers import AutoConfig
    arch = (AutoConfig.from_pretrained(model_name).architectures or [None])[0]
    from vllm.model_executor.models.registry import ModelRegistry
    return arch in set(ModelRegistry.get_supported_archs())


def build_lm_hf(model_name, tokenizer_name=None, any_adapter=None,
                max_model_len=4096):
    """Build the Hugging Face evaluation backend for unsupported vLLM models.

    The PEFT adapter is loaded at construction and cannot be swapped through
    set_adapter. Tasks, limits, and scoring definitions match the vLLM backend.
    A subclass overrides the read-only generation-token limit for longer answers.
    """
    from lm_eval.models.huggingface import HFLM

    # Detect reasoning support from the tokenizer.
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    _think_end = _tok.convert_tokens_to_ids(THINK_END_TOKEN)
    _reasoning = (_think_end is not None
                  and _think_end != getattr(_tok, "unk_token_id", None))

    # Allocate enough generation tokens for the reasoning trace and answer.
    _gen_toks = REASONING_GEN_TOKS if _reasoning else 512

    class _HFLMGen(HFLM):
        @property
        def max_gen_toks(self) -> int:
            return _gen_toks

    kw = {}
    if _reasoning:
        # Strip the reasoning trace before evaluating the answer.
        kw["think_end_token"] = _think_end

    lm = _HFLMGen(
        pretrained=model_name,
        tokenizer=tokenizer_name or model_name,
        peft=any_adapter or None,
        dtype="bfloat16",
        batch_size="auto",
        max_length=max_model_len if not _reasoning else max(max_model_len, _gen_toks + 2048),
        **kw,
    )
    if _reasoning:
        print(f"[lm_eval] {model_name} reasons: max_gen_toks={_gen_toks}, "
              f"think_end_token={_think_end}", flush=True)
    lm._baked_adapter = any_adapter
    return lm


def build_lm(model_name, tokenizer_name=None, any_adapter=None,
             gpu_memory_utilization=0.85, lora_rank=64, max_model_len=4096):
    """The vLLM engine, built once and reused across checkpoints.

    lora_local_path is what flips vLLM's enable_lora on, so an adapter path must
    be given at construction if any adapter will ever be used; which adapter is
    actually served is decided per call by set_adapter below.
    """
    # Restore tokenizer properties required by vLLM 0.7.3 with newer transformers.
    from utils.vllm_utils import _patch_vllm_tokenizer_compat
    _patch_vllm_tokenizer_compat()

    from lm_eval.models.vllm_causallms import VLLM
    return VLLM(
        pretrained=model_name, tokenizer=tokenizer_name or model_name,
        dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,   # IFEval generates up to 1280 new tokens
        # Set a generation budget for GSM8K; IFEval supplies its own task limit.
        max_gen_toks=512,
        max_lora_rank=lora_rank,       # adapters are r=64; lm_eval defaults to 16
        lora_local_path=any_adapter,
        batch_size="auto",
    )


_NEXT_LORA_ID = [0]


def set_adapter(lm, adapter):
    """Select an adapter or the base model for evaluation.

    The Hugging Face backend validates that the requested adapter matches the one
    loaded at construction. vLLM can switch adapters on the existing engine;
    each adapter receives a new integer ID to avoid reusing a cached adapter.
    """
    if hasattr(lm, "_baked_adapter"):
        if adapter != lm._baked_adapter:
            raise RuntimeError(
                f"HF backend was built for adapter {lm._baked_adapter!r} and "
                f"cannot switch to {adapter!r}; build a new lm instead")
        return
    if adapter is None:
        lm.lora_request = None
        return
    from vllm.lora.request import LoRARequest
    _NEXT_LORA_ID[0] += 1
    lm.lora_request = LoRARequest(f"ckpt{_NEXT_LORA_ID[0]}", _NEXT_LORA_ID[0], adapter)


_TM = [None]


def task_manager():
    """The task registry, built once: TaskManager walks every bundled task yaml on
    construction, which is seconds that would otherwise repeat per task per step."""
    if _TM[0] is None:
        from lm_eval.tasks import TaskManager
        _TM[0] = TaskManager()
    return _TM[0]


def _is_reasoning_lm(lm):
    """True if the model's tokenizer knows a </think> token.

    Detected from the tokenizer rather than a name list, so it needs no edit for
    a new reasoning model and is False for every model evaluated before now.
    """
    tok = getattr(lm, "tokenizer", None) or getattr(lm, "_tokenizer", None)
    if tok is None:
        return False
    try:
        tid = tok.convert_tokens_to_ids(THINK_END_TOKEN)
    except Exception:                                   # noqa: BLE001
        return False
    return tid is not None and tid != getattr(tok, "unk_token_id", None)


def evaluate(lm, adapter, mmlu_limit=MMLU_LIMIT, gsm8k_limit=GSM8K_LIMIT,
             ifeval_limit=IFEVAL_LIMIT):
    """Score one checkpoint on all three tasks. adapter=None scores the base model."""
    from lm_eval import simple_evaluate

    set_adapter(lm, adapter)
    reasoning = _is_reasoning_lm(lm)
    out = {}
    for task, limit in tasks(mmlu_limit, gsm8k_limit, ifeval_limit,
                             reasoning=reasoning):
        if not limit:
            continue
        # Disable chat templating for likelihood-based multiple-choice tasks on reasoning models.
        _chat = not (task in MULTIPLE_CHOICE_TASKS and reasoning)
        res = simple_evaluate(
            model=lm, tasks=[task],
            num_fewshot=0,             # explicit: no few-shot is introduced anywhere
            limit=limit,
            apply_chat_template=_chat,
            task_manager=task_manager(),
        )
        harvest(res, out)
    return out


def _worker(q, model_name, tokenizer_name, adapter, gpu_mem, lora_rank,
            mmlu_limit, gsm8k_limit, ifeval_limit):
    # Use soft locks before importing dataset libraries on filesystems without flock support.
    import filelock
    filelock.FileLock = filelock.SoftFileLock
    filelock.UnixFileLock = filelock.SoftFileLock
    print("[lm_eval] patched filelock -> SoftFileLock in worker", flush=True)
    try:
        if vllm_supports(model_name):
            lm = build_lm(model_name, tokenizer_name, any_adapter=adapter,
                          gpu_memory_utilization=gpu_mem, lora_rank=lora_rank)
        else:
            print(f"[lm_eval] {model_name} is not a vLLM 0.7.3 architecture; "
                  f"scoring through the HF backend", flush=True)
            lm = build_lm_hf(model_name, tokenizer_name, any_adapter=adapter)
        q.put(("ok", evaluate(lm, adapter, mmlu_limit, gsm8k_limit, ifeval_limit)))
    except Exception as exc:                       # noqa: BLE001
        import traceback
        q.put(("err", f"{exc}\n{traceback.format_exc()}"))


def run_lmeval_subprocess(model_name, tokenizer_name=None, lora_path=None,
                          gpu_memory_utilization=0.85, lora_rank=64,
                          mmlu_limit=MMLU_LIMIT, gsm8k_limit=GSM8K_LIMIT,
                          ifeval_limit=IFEVAL_LIMIT, timeout=7200):
    """Run the three evals in a spawned subprocess and return the metric dict.

    A subprocess for the same reason the existing utility eval uses one: vLLM
    cannot share the GPU with the training process's own model, and it does not
    release memory cleanly in-process.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    # Set before Process(), not inside the worker: spawn snapshots os.environ at
    # start, and `datasets` reads these at import, so setting them in the child
    # would already be too late. Restored immediately after the child is running,
    # so the training process keeps its own cache configuration.
    prev = {k: os.environ.get(k) for k in EVAL_ENV}
    os.environ.update(EVAL_ENV)
    try:
        proc = ctx.Process(target=_worker,
                           args=(q, model_name, tokenizer_name, lora_path,
                                 gpu_memory_utilization, lora_rank,
                                 mmlu_limit, gsm8k_limit, ifeval_limit))
        proc.start()
    finally:
        for _k, _v in prev.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v
    try:
        status, payload = q.get(timeout=timeout)
    except Exception:                              # noqa: BLE001
        proc.terminate(); proc.join(30)
        raise RuntimeError(f"lm_eval subprocess produced no result within {timeout}s")
    proc.join(120)
    if proc.is_alive():
        proc.terminate(); proc.join(30)
    if status != "ok":
        raise RuntimeError(f"lm_eval subprocess failed: {payload}")
    return payload
