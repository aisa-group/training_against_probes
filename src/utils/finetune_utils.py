"""
Utilities for probe-guided LoRA finetuning.
"""

import contextlib
import json
import os
import random
from typing import Optional

from utils.lora_utils import disable_lora

import torch
import torch.nn.functional as F
from datasets import load_dataset

HF_JB_DATASET = os.environ.get("PROBE_COMPLETIONS_DATASET", "lenalibon/jailbreak-judge-completions")
PROMPT_COL = "prompt"


# ──────────────────────────────── Data helpers ────────────────────────────────

def load_training_prompts(split_name: str, max_n: Optional[int] = None) -> list[str]:
    """Load raw user prompts for on-policy generation.

    Default source is a named split of lenalibon/jailbreak-judge-completions.

    A `split_name` of the form ``liars_bench:<config>:<side>`` instead pulls the
    on-policy DECEPTION prompts from Cadenza-Labs/liars-bench, where side is
    ``deceptive`` or ``honest`` (folded system+user prompt strings). This is how
    the deception finetuning configs point harmful_train_split / benign_train_split
    at Liar's Bench; see utils.jailbreak_datasets.load_liarsbench_prompts.
    """
    if split_name.startswith("liars_bench:"):
        from utils.jailbreak_datasets import load_liarsbench_prompts
        parts = split_name.split(":")
        if len(parts) != 3:
            raise ValueError(
                "liars_bench on-policy split must be "
                "'liars_bench:<config>:<deceptive|honest>', got "
                f"{split_name!r}")
        _, config, side = parts
        return load_liarsbench_prompts(config=config, side=side, max_n=max_n)
    ds = load_dataset(HF_JB_DATASET, split=split_name)
    prompts = list(ds[PROMPT_COL])
    if max_n is not None:
        prompts = prompts[:max_n]
    return prompts


def _first_user_turn(ex):
    """Pull the first user prompt out of whatever schema a chat dataset uses."""
    msgs = ex.get("messages") or ex.get("conversations") or ex.get("conversation")
    if isinstance(msgs, list) and msgs:
        first = msgs[0]
        if isinstance(first, dict):
            role = first.get("role") or first.get("from")
            content = first.get("content") or first.get("value")
            if content and (role is None or str(role).lower() in ("user", "human")):
                return content
    for key in ("prompt", "instruction", "question", "query", "text", "input"):
        v = ex.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


def load_kl_dataset(
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    max_n: int = 10_000,
) -> list[str]:
    """Load first user prompts for the KL anchor.

    Schema- and split-agnostic: UltraChat exposes ``train_sft`` with a ``messages``
    list, while other instruction sets use ``train`` and may use ``conversations``
    or a bare ``prompt`` column. Rather than hardcode one shape, try the likely
    splits in order and pull the first user turn out of whichever schema is present.
    """
    # A local JSON list of prompts, for datasets the container's `datasets` version
    # cannot parse. Dolci-Instruct-SFT uses the newer "List" feature type and raises
    # ValueError: Feature type 'List' not found on load, so its prompts are exported
    # once on the login node and read from disk here.
    if dataset_name.endswith(".json") and os.path.exists(dataset_name):
        with open(dataset_name) as fh:
            prompts = json.load(fh)
        if not prompts:
            raise ValueError(f"{dataset_name} contains no prompts")
        return prompts[:max_n]

    ds = None
    for split in ("train_sft", "train"):
        try:
            ds = load_dataset(dataset_name, split=split)
            break
        except Exception:
            continue
    if ds is None:
        raise ValueError(f"could not load a train split from {dataset_name}")
    prompts: list[str] = []
    for ex in ds:
        p = _first_user_turn(ex)
        if p:
            prompts.append(p)
        if len(prompts) >= max_n:
            break
    if not prompts:
        raise ValueError(f"no usable prompts found in {dataset_name}; "
                         f"columns were {list(ds.features)}")
    return prompts


def load_math_kl_dataset(
    dataset_name: str = "EleutherAI/hendrycks_math",
    max_n: int = 1_000,
) -> list[str]:
    """Load MATH problem statements or a local JSON prompt list for KL anchoring.

    MATH is separate from the GSM8K utility benchmark. Dataset configurations are
    loaded per subject, with a fixed subject list as a fallback when Hub lookup
    is unavailable. Raises ValueError if no prompts can be loaded.
    """
    # Load local JSON prompt pools without requiring Hub access.
    if dataset_name.endswith(".json") and os.path.exists(dataset_name):
        with open(dataset_name) as fh:
            prompts = json.load(fh)
        if not prompts:
            raise ValueError(f"{dataset_name} contains no prompts")
        return prompts[:max_n]

    from datasets import get_dataset_config_names

    MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry",
                     "intermediate_algebra", "number_theory", "prealgebra",
                     "precalculus"]
    try:
        configs = get_dataset_config_names(dataset_name) or MATH_SUBJECTS
    except Exception:
        configs = MATH_SUBJECTS

    prompts: list[str] = []
    errors = []
    for cfg in configs:
        try:
            ds = load_dataset(dataset_name, cfg, split="train")
        except Exception as exc:
            errors.append(f"{cfg}: {type(exc).__name__} {str(exc)[:80]}")
            continue
        for ex in ds:
            q = ex.get("problem") or ex.get("question")
            if q:
                prompts.append(q)
            if len(prompts) >= max_n:
                return prompts
    if not prompts:
        raise ValueError(
            f"maths KL pool is empty for {dataset_name}; tried configs {configs}. "
            f"Errors: {errors[:3]}"
        )
    return prompts


def sample_batch(pool: list[str], batch_size: int) -> list[str]:
    """Random sample without replacement; repeats pool if smaller than batch."""
    n = len(pool)
    if n >= batch_size:
        return random.sample(pool, batch_size)
    reps = (batch_size + n - 1) // n
    return (pool * reps)[:batch_size]


# ─────────────────────────────── Tokenisation ────────────────────────────────

# Process-wide reasoning setting, used by chat templates that support enable_thinking.
_ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0").lower() in ("1", "true", "yes")


def set_enable_thinking(flag: bool) -> None:
    """Set the reasoning switch for this process. Called from finetune.py once
    the config is read, so a single config key controls generation, the anchor
    and every downstream eval consistently."""
    global _ENABLE_THINKING
    _ENABLE_THINKING = bool(flag)


def apply_chat_template(prompts: list[str], tokenizer,
                        enable_thinking: Optional[bool] = None) -> list[str]:
    """Wrap raw user prompts in the model's chat template (ready-to-generate).

    `enable_thinking` controls Qwen3's <think> block: False closes it
    immediately so the model answers directly, True lets it reason. The kwarg is
    ignored by templates that do not reference it (Llama, Mistral), so this is a
    no-op for every non-reasoning model. Defaults to the process-wide switch set
    by set_enable_thinking(), which itself defaults to False.
    """
    flag = _ENABLE_THINKING if enable_thinking is None else enable_thinking
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=flag,
        )
        for p in prompts
    ]


def tokenize_prompt_completion_batch(
    prompts: list[str],
    completions: list[str],
    tokenizer,
    max_length: int = 1024,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenise chat-formatted prompt + completion pairs.

    Returns
    -------
    input_ids      : (batch, seq)
    attention_mask : (batch, seq)
    completion_mask: (batch, seq)  True = completion (generation) token
    """
    formatted = apply_chat_template(prompts, tokenizer)
    combined = [fp + c for fp, c in zip(formatted, completions)]

    # Prompt lengths (needed to build completion mask)
    prompt_lengths = [
        len(tokenizer(fp, add_special_tokens=False).input_ids)
        for fp in formatted
    ]

    enc = tokenizer(
        combined,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    seq_len = input_ids.shape[1]
    indices = torch.arange(seq_len, device=device).unsqueeze(0)          # (1, seq)
    prompt_end = torch.tensor(prompt_lengths, device=device).unsqueeze(1)  # (batch, 1)
    completion_mask = (indices >= prompt_end) & attention_mask.bool()

    return input_ids, attention_mask, completion_mask


THINK_END = "</think>"


def answer_only_text(text: str, think_end: str = THINK_END) -> str:
    """Return text after the last thinking-block terminator.

    For reasoning completions with no closing terminator, return an empty string.
    Use only for reasoning runs; ordinary completions may have no terminator.
    """
    if think_end in text:
        return text.rsplit(think_end, 1)[1]
    return ""


def answer_only_mask(input_ids, completion_mask, tokenizer):
    """Restrict the completion loss mask to tokens after the thinking block.

    Detector loss scores the answer, while the KL anchor can include both reasoning
    and answer tokens. Attention remains unchanged; only scored positions differ.
    Returns (mask, n_without_answer). Unclosed thinking blocks contribute no answer
    tokens, and the count allows callers to track these cases.
    """
    ids = tokenizer.convert_tokens_to_ids(THINK_END)
    if ids is None or ids == getattr(tokenizer, "unk_token_id", None):
        # Not a reasoning tokenizer: nothing to strip.
        return completion_mask, 0

    is_end = (input_ids == ids) & completion_mask
    if not bool(is_end.any()):
        return torch.zeros_like(completion_mask), int(completion_mask.shape[0])

    seq = input_ids.shape[1]
    idx = torch.arange(seq, device=input_ids.device).unsqueeze(0)
    # Last </think> per row; -1 where the row has none.
    last = torch.where(is_end, idx, torch.full_like(idx, -1)).max(dim=1).values
    has = last >= 0
    mask = completion_mask & (idx > last.unsqueeze(1)) & has.unsqueeze(1)
    return mask, int((~has).sum())


def thinking_length_backward(model, input_ids, attention_mask, think_mask,
                             think_end_id, penalty, micro_batch=8):
    """Small length penalty on the reasoning trace: encourage </think> as the next
    token throughout the thinking region, biasing the model toward ending its trace
    sooner (shorter chain of thought) WITHOUT the detector ever scoring the
    reasoning -- the probe stays answer-only. Differentiable surrogate for a length
    reward (cf. the RL length-penalty literature, e.g. Kimi/R1 cosine reward,
    ThinkPrune): minimise the cross-entropy toward </think> at every thinking
    position, i.e. raise p(</think> | prefix), so the trace terminates earlier.

    Backprops internally, micro-batched over the batch (one backward per micro,
    each scaled by its share of thinking positions), so the accumulated gradient
    equals  penalty * mean_{thinking positions} [-log p(</think>)]. Returns that
    mean (penalty-free) for logging; 0.0 if there are no thinking tokens.
    """
    think_mask = think_mask.bool()
    total = int(think_mask.sum().item())
    if total == 0 or think_end_id is None:
        return 0.0
    B = input_ids.shape[0]
    val = 0.0
    for s in range(0, B, micro_batch):
        e = min(s + micro_batch, B)
        tm = think_mask[s:e]
        if not bool(tm.any()):
            continue
        logits = model(input_ids=input_ids[s:e],
                       attention_mask=attention_mask[s:e]).logits    # (b, S, V)
        # float32 for the softmax: in bf16 the log-prob of </think> at a
        # mid-thinking position underflows to -inf (the model assigns it ~0), so
        # the cross-entropy becomes inf and poisons the gradient.
        sel = logits[tm].float()                                     # (n, V)
        tgt = torch.full((sel.shape[0],), think_end_id,
                         device=sel.device, dtype=torch.long)
        loss = F.cross_entropy(sel, tgt, reduction="sum") / total
        if not torch.isfinite(loss):
            del logits, sel, loss
            continue
        (penalty * loss).backward()
        val += float(loss)
        del logits, sel, loss
    return val


def load_kl_rollouts(path: str, max_n: int = 1_000
                          ) -> list[tuple[str, str]]:
    """Load fixed (prompt, base-model completion) pairs for the KL anchor.

    Generate this reference text once with probes/build_kl_rollouts.py and reuse it
    throughout finetuning. Missing files and empty reference pools raise ValueError.
    """
    if not os.path.exists(path):
        raise ValueError(f"rollout anchor file not found: {path}; "
                         f"build it with probes/build_kl_rollouts.py")
    with open(path) as fh:
        rows = json.load(fh)
    pairs = [(r["prompt"], r["completion"]) for r in rows
             if r.get("prompt") and r.get("completion")]
    if not pairs:
        raise ValueError(f"{path} contains no usable (prompt, completion) pairs")
    return pairs[:max_n]


def tokenize_kl_batch(
    prompts: list[str],
    tokenizer,
    max_length: int = 512,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise KL prompts (chat-formatted, no completion). Returns (input_ids, attention_mask)."""
    formatted = apply_chat_template(prompts, tokenizer)
    enc = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ─────────────────────────────── Loss helpers ────────────────────────────────

def compute_probe_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    probes: dict,
    probe_loss_layers: list[int],
    loss_type: str = "bce",
    hinge_margin: float = 1.0,
    pooling: str = "mean",
) -> torch.Tensor:
    """
    Probe loss that pushes probe scores toward 0 on *all* inputs
    (both harmful and benign on-policy completions).

    Gradient flow: loss → probe (frozen weights) → model activations → LoRA weights.

    Parameters
    ----------
    model            : PEFT model with LoRA active, in train() mode.
    input_ids        : (batch, seq)
    attention_mask   : (batch, seq)
    completion_mask  : (batch, seq)  True = generation token
    probes           : {layer_int: Probe}  all weights frozen, eval() mode
    probe_loss_layers: Layer indices that count toward the normalised loss.
    loss_type        : 'bce' (default) or 'hinge'. Hinge uses margin and
                       saturates exactly at the boundary — see
                       probes.probe_archs.compute_loss docstring.
    hinge_margin     : margin parameter for hinge loss. Default 1.0.

    Returns
    -------
    Scalar loss tensor with gradient attached.
    """
    # Import here to keep the module importable without sys.path gymnastics
    import sys, os as _os
    _probe_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "probes")
    if _probe_dir not in sys.path:
        sys.path.insert(0, _probe_dir)
    from probe_archs import compute_loss, mean_aggregator

    # Skip the language-model head because probe loss only requires hidden states.
    backbone = model.base_model.model.model if hasattr(model, "base_model") else model.model
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq, hidden)
    hidden_states = outputs.hidden_states

    aggregator = mean_aggregator()
    batch_size = input_ids.shape[0]
    total_loss = torch.zeros(1, device=input_ids.device, dtype=torch.float32).squeeze()

    for layer_idx in probe_loss_layers:
        # +1: hidden_states[0] = embedding layer output
        acts = hidden_states[layer_idx + 1].float()      # (batch, seq, hidden)
        probe = probes[layer_idx].to(acts.device)

        # With device_map="auto" the backbone is sharded — acts for late layers
        # lives on cuda:1/2/3 while completion_mask and labels were built on
        # the input device (cuda:0). Move them to acts.device per layer.
        mask_dev = completion_mask.to(acts.device)
        if pooling == "token":
            # Per-token loss, then a masked mean, exactly the ordering
            # compute_polytope_lora_loss uses. The default pools the token scores
            # FIRST and takes one loss per sequence, which lets a few very harmful
            # tokens average out against many benign ones; scoring per token does
            # not. Isolating that ordering is the point of this option.
            z = probe(acts, padding_mask=None).squeeze(-1)      # (batch, seq)
            if loss_type == "hinge":
                per_tok = torch.clamp(hinge_margin + z, min=0.0)
            else:
                per_tok = F.binary_cross_entropy_with_logits(
                    z, torch.zeros_like(z), reduction="none")
            m = mask_dev.float()
            loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)
        else:
            # Target 0 for all inputs: push probe to predict "benign"
            labels = torch.zeros(batch_size, device=acts.device)
            loss = compute_loss(
                probe, acts, labels, mask=mask_dev, aggregator=aggregator,
                loss_type=loss_type, hinge_margin=hinge_margin,
            )
        # total_loss lives on input_ids.device; ensure we add on a consistent device.
        total_loss = total_loss + loss.to(total_loss.device)

    return total_loss / max(len(probe_loss_layers), 1)


def compute_polytope_lora_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    probes: dict,
    polytope_layers: list[int],
    margin: float = 1.0,
    mode: str = "sum_relu",
    lse_temp: float = 1.0,
    use_autocast: bool = True,
) -> torch.Tensor:
    """Polytope analogue of compute_probe_loss: push every on-policy completion
    INSIDE the safety polytope, with margin.

    Gradient flow: hinge -> frozen polytope -> model activations -> LoRA weights.

    Where the BCE version drives one scalar probe score toward 0, this drives
    all K facet violations below -margin. Two consequences that are the point of
    the whole polytope line of work:

      * It saturates. relu(margin + v_k) is exactly 0 once the completion is
        genuinely inside facet k with margin, so there is no unbounded
        "keep pushing past the boundary" gradient. K facets means K independent
        saturating constraints, so the region the LoRA has to reach is a
        K-dimensional intersection rather than a single half-space.
      * Benign completions that are already inside contribute exactly zero, so
        the term is a pure guardrail instead of the constant push the BCE
        version applies to harmful and benign alike.

    Parameters
    ----------
    model            : PEFT model with LoRA active, in train() mode.
    input_ids        : (batch, seq)
    attention_mask   : (batch, seq)
    completion_mask  : (batch, seq)  True = generation token
    probes           : {layer_int: PolytopeProbe}  frozen weights, eval() mode
    polytope_layers  : Layer indices that count toward the normalised loss.
    margin           : how far inside each facet a completion must sit.
    mode             : 'sum_relu' (default) sum_k relu(margin + v_k), as SaP
                           -- dense gradient on every facet; scale grows with K.
                       'mean_relu'          mean_k relu(margin + v_k)
                           -- same zero set, K-independent scale.
                       'max_relu'           relu(margin + max_k v_k)
                           -- worst facet only; the gradient reaches just the
                              argmax facet, so it plays whack-a-mole across the
                              K walls. Ablation only.
                       'lse'                relu(margin + T*logsumexp(v/T))
                           -- smooth interpolation between the two.
    lse_temp         : temperature T for mode='lse'.
    use_autocast     : run the concept encoder in bf16. The encoder output f is
                       (batch, seq, feature_dim) and is retained for the
                       backward, so this roughly halves the dominant activation
                       cost. Reductions still happen in fp32.

    Returns
    -------
    Scalar loss tensor with gradient attached.
    """
    import sys, os as _os
    _probe_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "probes")
    if _probe_dir not in sys.path:
        sys.path.insert(0, _probe_dir)

    # Skip the language-model head because probe loss only requires hidden states.
    backbone = model.base_model.model.model if hasattr(model, "base_model") else model.model
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states

    total_loss = torch.zeros(1, device=input_ids.device, dtype=torch.float32).squeeze()

    for layer_idx in polytope_layers:
        # +1: hidden_states[0] = embedding layer output
        acts = hidden_states[layer_idx + 1]
        probe = probes[layer_idx].to(acts.device)
        mask_dev = completion_mask.to(acts.device)

        # With device_map="auto" the backbone is sharded, so acts for late
        # layers can live on a different device than the inputs.
        if use_autocast and acts.device.type == "cuda":
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            v = probe.violations(acts, padding_mask=mask_dev)  # (batch, seq, K)

        v = v.float()
        if mode == "sum_relu":
            # Sum hinge penalties across facets. The loss scale grows with K, affecting its
            # weight relative to the KL term.
            per_tok = torch.relu(margin + v).sum(dim=-1)
        elif mode == "mean_relu":
            # Average hinge penalties across facets for a scale independent of K.
            per_tok = torch.relu(margin + v).mean(dim=-1)
        elif mode == "max_relu":
            per_tok = torch.relu(margin + v.amax(dim=-1))
        elif mode == "bce":
            # BCE targets the benign side of every facet and continues applying pressure
            # after the hinge margin would be satisfied.
            per_tok = F.binary_cross_entropy_with_logits(
                v, torch.zeros_like(v), reduction="none").sum(dim=-1)
        elif mode == "lse":
            per_tok = torch.relu(margin + lse_temp * torch.logsumexp(v / lse_temp, dim=-1))
        else:
            raise ValueError(f"unknown polytope lora loss mode: {mode!r}")

        m = mask_dev.float()
        denom = m.sum().clamp(min=1.0)
        loss = (per_tok * m).sum() / denom
        total_loss = total_loss + loss.to(total_loss.device)

    return total_loss / max(len(polytope_layers), 1)


def rollout_kl_backward(model, pairs, tokenizer, *, penalty, max_length,
                        micro_batch, chunk_size, device):
    """KL against frozen teacher completions, micro-batched, backward included.

    `pairs` are (prompt, completion) where the completion was written once by the
    un-finetuned model and never regenerated. Scores only the completion tokens,
    while the model still attends to the prompt that conditions them.

    Micro-batched because prompt+completion is roughly twice a bare prompt, and
    compute_kl_loss runs two full-vocab forwards; splitting keeps the number of
    prompts per step unchanged at the peak memory of the smaller batch. Each part
    is weighted by its share of scored tokens, so the result is the token-mean
    over the whole batch and not a mean of means. backward() runs per micro-batch
    so each graph is freed before the next is built; the grads accumulate into the
    same buffers, which is what makes this identical to one large batch.

    Returns the (unweighted by penalty) mean KL as a float, for logging.
    """
    micro = micro_batch or len(pairs)
    chunks = [pairs[i:i + micro] for i in range(0, len(pairs), micro)]

    batches = []
    for ch in chunks:
        ids, attn, comp = tokenize_prompt_completion_batch(
            prompts=[q for q, _ in ch], completions=[c for _, c in ch],
            tokenizer=tokenizer, max_length=max_length, device=device,
        )
        batches.append((ids, attn, comp, float(comp.sum())))

    grand = sum(b[3] for b in batches)
    if grand <= 0:
        raise ValueError(
            "rollout KL batch has no completion tokens, so the anchor would be a "
            "no-op. max_length is probably too short to leave room for the "
            "completion after the prompt")

    total = 0.0
    for ids, attn, comp, w in batches:
        part = compute_kl_loss(model=model, input_ids=ids, attention_mask=attn,
                               chunk_size=chunk_size, loss_mask=comp)
        share = w / grand
        total += float(part) * share
        (penalty * share * part).backward()
        del part
    return total


def compute_kl_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int = 64,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute token-level KL(p_base || p_adapted) over scored positions.

    Base logits are obtained by temporarily disabling LoRA adapters. Sequence
    chunking limits memory use while preserving the global token average.

    Args:
        model: PEFT model with LoRA adapters.
        input_ids: Token IDs with shape (batch, sequence).
        attention_mask: Positions visible to the model.
        chunk_size: Sequence positions processed per chunk.
        loss_mask: Positions included in the KL average; defaults to attention_mask.

    Separate attention and loss masks allow scoring completion tokens while the
    model still attends to the prompt. The loss mask determines both numerator
    weights and the denominator.

    Returns:
        Scalar KL tensor with gradients through the adapted logits.
    """
    seq_len = input_ids.shape[1]
    if loss_mask is None:
        loss_mask = attention_mask

    # ── Base model logits (LoRA disabled, no gradient) ────────────────────────
    with disable_lora(model):
        with torch.no_grad():
            base_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits  # (batch, seq, vocab)

    # ── Adapted model logits (LoRA active, with gradient) ─────────────────────
    adapted_logits = model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits  # (batch, seq, vocab) — gradient attached

    # With device_map="auto" the lm_head's output (logits) lives on the last
    # shard's GPU, while attention_mask was built on the input device. Operate
    # on logits.device to avoid cross-device arithmetic errors.
    logits_device = adapted_logits.device
    total_tokens = loss_mask.float().to(logits_device).sum().clamp(min=1.0)
    kl_total = torch.zeros(1, device=logits_device, dtype=torch.float32).squeeze()

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        # Compute log-softmax for this sequence chunk
        base_lp = F.log_softmax(base_logits[:, start:end].float(), dim=-1).detach()
        adapted_lp = F.log_softmax(adapted_logits[:, start:end].float(), dim=-1)

        # KL(base ‖ adapted) = Σ_v p_base(v) · (log p_base(v) − log p_adapted(v))
        # F.kl_div(input=log_q, target=p) = Σ p·(log p − log q)   [no reduction]
        kl_chunk = F.kl_div(
            adapted_lp, base_lp.exp(), reduction="none", log_target=False
        ).sum(-1)  # (batch, chunk)

        mask_chunk = loss_mask[:, start:end].float().to(logits_device)
        kl_total = kl_total + (kl_chunk * mask_chunk).sum()

    return kl_total / total_tokens


# ─────────────────────────── vLLM on-policy generation ───────────────────────

def generate_on_policy_vllm(
    formatted_prompts: list[str],
    model_name: str,
    tokenizer_name: Optional[str],
    lora_path: Optional[str],
    max_new_tokens: int = 200,
    gpu_memory_utilization: float = 0.85,
    lora_rank: int = 64,
) -> list[str]:
    """
    Generate on-policy completions via vLLM (greedy decoding).

    A fresh vLLM instance is created and destroyed on every call to avoid
    GPU-memory conflicts with the HF training model.  The caller is responsible
    for moving the HF model to CPU before this call and back afterwards.

    Parameters
    ----------
    formatted_prompts      : Already chat-formatted strings (add_generation_prompt=True).
    model_name             : Base HuggingFace model ID.
    tokenizer_name         : Override tokenizer (None → model's own tokenizer).
    lora_path              : Path to saved LoRA adapter directory.  None → base model.
    max_new_tokens         : Maximum tokens to generate.
    gpu_memory_utilization : Fraction of GPU VRAM vLLM may use.
    lora_rank              : max_lora_rank passed to vLLM (must be ≥ adapter rank).

    Returns
    -------
    List of completion strings (prompt not included).
    """
    from utils.vllm_utils import generate_vllm_subprocess

    return generate_vllm_subprocess(
        formatted_prompts=formatted_prompts,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        lora_path=lora_path,
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        lora_rank=lora_rank,
    )


# ─────────────────────────── Probe score extraction ──────────────────────────

@torch.no_grad()
def extract_probe_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    probes: dict,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """
    Run model forward (no grad) and return per-layer probe probabilities.

    Returns
    -------
    {layer_idx: (batch,) probability tensor on CPU}
    """
    import sys, os as _os
    _probe_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "probes")
    if _probe_dir not in sys.path:
        sys.path.insert(0, _probe_dir)
    from probe_archs import mean_aggregator

    model.eval()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states
    aggregator = mean_aggregator()
    device = input_ids.device

    scores: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        acts = hidden_states[layer_idx + 1].float().to(device)  # (batch, seq, hidden)
        probe = probes[layer_idx].to(device).eval()
        padding_mask = completion_mask.to(device)

        # Raw probe output: (batch, seq, nhead)
        if aggregator.needs_q and hasattr(probe, "forward_qv"):
            q, v = probe.forward_qv(acts, padding_mask=padding_mask)
        else:
            q, v = None, probe(acts, padding_mask=padding_mask)

        logits = aggregator(v, padding_mask, q=q)        # (batch,)
        scores[layer_idx] = torch.sigmoid(logits).cpu()

    return scores
