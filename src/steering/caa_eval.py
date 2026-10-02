"""Contrastive activation addition (CAA) steering baseline.

Training-free comparison arm: the base model's weights stay fixed and a constant
safety direction is added to the residual stream at one layer, scaled by a
coefficient that is swept instead of a training step.

Direction. Mean layer output over the completion tokens of each example in the
paired BeaverTails set (build_beavertails_paired_probe_datasets, the same data and
n_train the finetuning arms use), then mean(safe) - mean(harmful), normalised to
unit length. The coefficient is a fraction of the mean completion-token residual
norm at that layer: h <- h + coef * ref_norm * d.

Sweep. coef = 0, step, 2*step, ... until aggregate utility falls below
util_floor * (coef 0 utility); the first sub-floor point is kept. --alphas gives
an explicit grid instead.

Scoring. Utility through lm-evaluation-harness on the HF backend (MMLU, GSM8K,
IFEval via build_lm_hf + evaluate + aggregate_utility), so a forward hook can
reach the residual stream. StrongREJECT on the first 100 JailbreakBench prompts,
greedy, 256 new tokens, judged after the model is released (the judge runs in
its own vLLM process).

Output: $NM_ROOT/steering_eval/steer_caa_n<N>.json unless --out is given. An
existing output is resumed: points with a StrongREJECT score are reused.

    python src/steering/caa_eval.py --n-train 750
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SRC, "probes"))
sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from utils.paths import NM_ROOT  # noqa: E402

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LAYER = 19
SR_N = 100
SR_MAX_NEW = 256


def _hidden_completion(model, tok, prompts, completions, layer, batch=8, max_len=512):
    """Output of decoder layer `layer`, averaged over completion tokens.

    Returns (per-example means [N, d], per-token norms [total completion tokens]).
    """
    pooled, norms = [], []
    for i in range(0, len(prompts), batch):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + batch]]
        p_ids = tok(chats, add_special_tokens=False)["input_ids"]
        full = [c + comp for c, comp in zip(chats, completions[i:i + batch])]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            # hidden_states[0] is the embedding, so layer L's output is index L + 1.
            h = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
        am = enc["attention_mask"].bool()
        for r in range(h.shape[0]):
            # Index valid positions explicitly so left and right padding both work.
            pos = am[r].nonzero().flatten()
            plen = min(len(p_ids[r]), pos.numel() - 1)
            comp = pos[plen:]
            hr = h[r][comp] if comp.numel() else h[r][pos][-1:]
            pooled.append(hr.mean(0).cpu())
            norms.append(hr.norm(dim=-1).cpu())
    return torch.stack(pooled), torch.cat(norms)


def caa_direction(model, tok, harmful_p, harmful_c, safe_p, safe_c, layer=LAYER):
    """Return (mean(safe) - mean(harmful), mean completion-token residual norm)."""
    hh, nh = _hidden_completion(model, tok, harmful_p, harmful_c, layer)
    hs, ns = _hidden_completion(model, tok, safe_p, safe_c, layer)
    d = hs.mean(0) - hh.mean(0)
    ref = float(torch.cat([nh, ns]).mean())
    return d.numpy(), ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-train", type=int, required=True,
                    help="paired BeaverTails pairs used to fit the direction")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: $NM_ROOT/steering_eval/steer_caa_n<N>.json)")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--util-floor", type=float, default=0.8)
    ap.add_argument("--max-alpha", type=float, default=1.5)
    ap.add_argument("--alphas", type=float, nargs="*", default=None,
                    help="explicit coefficient grid; disables the adaptive stop")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from asr.downstream_eval import aggregate_utility
    from asr.judges import score_strongreject_batch
    from utils.finetune_utils import apply_chat_template
    from utils.jailbreak_datasets import (build_beavertails_paired_probe_datasets,
                                          load_jailbreakbench)
    from utils.lmeval_utils import build_lm_hf, evaluate

    base_model, layer_idx = args.model, args.layer
    stem = f"steer_caa_n{args.n_train}"
    out = args.out or (NM_ROOT / "steering_eval" / f"{stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    # pos = harmful, neg = safe; row i of each shares a prompt.
    pos, neg, _ = build_beavertails_paired_probe_datasets(n_train=args.n_train)
    harmful_p = [r["prompt"] for r in pos]
    harmful_c = [r["completion"] for r in pos]
    safe_p = [r["prompt"] for r in neg]
    safe_c = [r["completion"] for r in neg]

    # One HF-backend harness; lm.model is the module the hook attaches to.
    lm = build_lm_hf(base_model, base_model, any_adapter=None)
    model = lm.model
    model.eval()
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    d, ref = caa_direction(model, tok, harmful_p, harmful_c, safe_p, safe_c, layer_idx)
    d = d / (np.linalg.norm(d) + 1e-9)
    dt = torch.tensor(d, dtype=torch.bfloat16, device="cuda")
    print(f"[{stem}] direction ready, ref residual norm = {ref:.2f}, "
          f"|d|=1, layer {layer_idx}", flush=True)

    jbb = load_jailbreakbench()[:SR_N]
    jbb_fmt = apply_chat_template(jbb, tok)
    layers = model.model.layers

    def _write(points):
        merged = list(done.values()) + points
        merged.sort(key=lambda p: p["coef"])
        out.write_text(json.dumps({
            "stem": stem, "direction": "caa", "n_train": args.n_train,
            "base_model": base_model, "layer": layer_idx, "scorer": "lm_eval",
            "ref_norm": ref, "sr_n": SR_N, "points": merged}, indent=2))

    # Resume: reuse judged points (keyed by rounded coef); utility-only points rerun.
    done: dict[float, dict] = {}
    if out.exists():
        try:
            for p in json.load(open(out)).get("points", []):
                if p.get("sr") is not None:
                    done[round(float(p["coef"]), 4)] = p
        except (OSError, ValueError):
            pass

    # Phase 1 (model loaded): utility and StrongREJECT generations per coefficient.
    # Phase 2 (model released): judge, since the judge's vLLM process needs the VRAM.
    def _eval_alpha(alpha):
        mag = alpha * ref
        handle = None
        if alpha != 0.0:
            def hook(_m, _i, o, m=mag):
                h = o[0] if isinstance(o, tuple) else o
                h = h + m * dt
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = layers[layer_idx].register_forward_hook(hook)
        try:
            u = aggregate_utility(evaluate(lm, None))
            gen = []
            for i in range(0, len(jbb_fmt), 8):
                enc = tok(jbb_fmt[i:i + 8], return_tensors="pt", padding=True,
                          add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=SR_MAX_NEW, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                gen += tok.batch_decode(o[:, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
        finally:
            if handle is not None:
                handle.remove()
        torch.cuda.empty_cache()
        return mag, u, gen

    def _partial(swept):
        _write([{"coef": aa, "magnitude": m, "mmlu": uu["mmlu"], "gsm8k": uu["gsm8k"],
                 "ifeval_avg": uu["ifeval_avg"], "aggregate": uu["aggregate"], "sr": None}
                for aa, m, uu, _g in swept])

    swept = []            # (alpha, magnitude, utility, generations) for new alphas
    base_agg = None
    if args.alphas is not None:
        for a in [round(float(x), 4) for x in args.alphas]:
            if a in done:
                continue
            mag, u, gen = _eval_alpha(a)
            swept.append((a, mag, u, gen))
            print(f"[{stem}] alpha={a:.2f}  agg={u['aggregate']:.3f}  (SR pending)", flush=True)
            _partial(swept)
    else:
        a = 0.0
        while a <= args.max_alpha + 1e-9:
            ar = round(a, 4)
            if ar in done:
                agg = done[ar]["aggregate"]
            else:
                mag, u, gen = _eval_alpha(ar)
                swept.append((ar, mag, u, gen))
                agg = u["aggregate"]
                print(f"[{stem}] alpha={ar:.2f}  agg={agg:.3f}  (SR pending)", flush=True)
                _partial(swept)
            if base_agg is None:
                base_agg = agg
            if ar > 0 and base_agg and agg < args.util_floor * base_agg:
                print(f"[{stem}] utility {agg:.3f} < {args.util_floor:.0%} of base "
                      f"{base_agg:.3f} at alpha={ar:.2f}; stopping", flush=True)
                break
            a += args.step

    # Drop every reference to the model before the judge starts.
    del lm, layers, dt
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{stem}] utility+generation done; released the model, judging SR", flush=True)

    points = []
    for alpha, mag, u, gen in swept:
        sr = float(np.mean(score_strongreject_batch(jbb, gen)))
        points.append({"coef": alpha, "magnitude": mag,
                       "mmlu": u["mmlu"], "gsm8k": u["gsm8k"],
                       "ifeval_avg": u["ifeval_avg"], "aggregate": u["aggregate"],
                       "sr": sr})
        print(f"[{stem}] alpha={alpha:.2f}  SR={sr:.3f}", flush=True)
        _write(points)

    print(f"[{stem}] wrote {out}  ({len(points)} points)", flush=True)


if __name__ == "__main__":
    main()
