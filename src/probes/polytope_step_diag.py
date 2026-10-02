"""Compute training-step diagnostics on a fixed activation set.

Reports facet geometry (effective rank, cosine similarity, duplicate pairs),
polytope membership, constraint violations, and held-out detection AUROC.
The activation set is captured at step 0 for comparisons across training steps."""
import json
import os
import sys

import torch

_P = os.path.dirname(os.path.abspath(__file__))
_S = os.path.dirname(_P)
for p in (_P, _S):
    if p not in sys.path:
        sys.path.insert(0, p)


def _auroc(scores, labels):
    import numpy as np
    o = np.argsort(scores); r = np.empty(len(scores), float); r[o] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def polytope_geometry(probe) -> dict:
    """Measure facet rank, similarity, and weight norms."""
    phi = probe.phi.detach().float()
    n = phi.norm(dim=1)
    u = phi / n.clamp(min=1e-9).unsqueeze(1)
    C = (u @ u.t()).cpu()
    K = C.shape[0]
    if K < 2:
        return {"eff_rank": 1.0, "cos_max": 0.0, "cos_mean": 0.0, "dup_pairs": 0,
                "phi_norm_min": float(n.min()), "phi_norm_max": float(n.max())}
    off = C[~torch.eye(K, dtype=torch.bool)]
    s = torch.linalg.svdvals(phi).cpu()
    return {
        "eff_rank": float(s.sum() ** 2 / (s ** 2).sum()),   # participation ratio
        "cos_max": float(off.max()), "cos_mean": float(off.mean()),
        "dup_pairs": int((off > 0.9).sum()) // 2,
        "phi_norm_min": float(n.min()), "phi_norm_max": float(n.max()),
    }


@torch.no_grad()
def step_diagnostics(detector, layers, acts_by_layer, labels, mask, device="cuda") -> dict:
    """Full per-step diagnostic dict, ready for wandb.log and a JSON dump.

    acts_by_layer: {layer_idx: (N, S, D) float tensor} captured once at step 0.
    """
    from polytope_stats import global_polytope_stats

    out = {}
    for layer in layers:
        probe = detector[layer]
        if not hasattr(probe, "violations"):      # not a polytope: skip
            continue
        acts = acts_by_layer[layer]
        st = global_polytope_stats(probe, acts, labels, mask, device=device)
        gm = polytope_geometry(probe)

        scores = []
        for i in range(0, acts.shape[0], 32):
            a, m = acts[i:i + 32].to(device), mask[i:i + 32].to(device)
            v = probe.violations(a).amax(-1).masked_fill(~m, 0.0)
            scores.append((v.sum(1) / m.float().sum(1).clamp(min=1)).float().cpu())
        auc = _auroc(torch.cat(scores).numpy(), labels.numpy())

        p = f"polytope/L{layer}/"
        out.update({
            p + "eff_rank": gm["eff_rank"],
            p + "eff_rank_frac": gm["eff_rank"] / max(probe.num_facets, 1),
            p + "cos_max": gm["cos_max"],
            p + "cos_mean": gm["cos_mean"],
            p + "dup_pairs": gm["dup_pairs"],
            p + "phi_norm_min": gm["phi_norm_min"],
            p + "phi_norm_max": gm["phi_norm_max"],
            p + "phi_norm_ratio": gm["phi_norm_max"] / max(gm["phi_norm_min"], 1e-9),
            p + "frac_harmful_outside": st["frac_harmful_outside"],
            p + "frac_benign_outside": st["frac_benign_outside"],
            # mean number of violated constraints
            p + "facets_violated_per_example": st["avg_facets_violated"],
            p + "facets_ever_violated": st["facets_ever_violated"],
            p + "facet_entropy_bits": st["facet_entropy_bits"],
            p + "auroc": auc,
        })
    return out


@torch.no_grad()
def facet_category_map(probe, acts, categories, mask, device="cpu", chunk=32) -> dict:
    """Which facet fires on which kind of harm, and how concentrated is it?

    The premise of a polytope over a single hyperplane is that different facets
    capture different *kinds* of harm -- SaP's Figure 5 shows one facet for
    kidnapping, another for sexual content, another for bullying. If that is
    real, a facet's violation rate should be concentrated on a few categories
    rather than uniform across them.

    We use the 10 JailbreakBench categories (Disinformation, Fraud/Deception,
    Malware/Hacking, Privacy, Physical harm, ...), which are already labelled in
    our eval set and cost nothing extra.

    Returns, per facet, the violation rate on each category plus a specialisation
    score = 1 - H(rate distribution)/log2(n_categories). 1.0 means the facet
    fires on exactly one category; 0.0 means it fires uniformly on all of them
    and is doing nothing category-specific.
    """
    import math
    cats = sorted(set(categories))
    ci = {c: i for i, c in enumerate(cats)}
    K, C = probe.num_facets, len(cats)
    hits = torch.zeros(K, C)
    seen = torch.zeros(C)

    for i in range(0, acts.shape[0], chunk):
        a, m = acts[i:i + chunk].to(device), mask[i:i + chunk].to(device)
        v = probe.violations(a).float().masked_fill(~m[..., None], -1e4).amax(1)  # (b, K)
        viol = (v > 0).float().cpu()
        for j, c in enumerate(categories[i:i + chunk]):
            hits[:, ci[c]] += viol[j]
            seen[ci[c]] += 1

    rate = hits / seen.clamp(min=1).unsqueeze(0)                 # (K, C)
    tot = rate.sum(dim=1, keepdim=True).clamp(min=1e-9)
    p_ = rate / tot
    ent = -(p_ * (p_ + 1e-10).log2()).sum(dim=1)                 # (K,)
    spec = 1.0 - ent / math.log2(max(C, 2))
    live = rate.sum(dim=1) > 0
    return {
        "categories": cats,
        "rate": rate.tolist(),
        "specialisation": spec.tolist(),
        "mean_specialisation": float(spec[live].mean()) if bool(live.any()) else 0.0,
        "n_live_facets": int(live.sum()),
        "top_category_per_facet": [cats[int(r.argmax())] if bool(l) else None
                                   for r, l in zip(rate, live)],
    }


def dump_step_diagnostics(stats: dict, step_ckpt_path: str) -> None:
    """Persist alongside the step's adapter + probe snapshot, so any of this can
    be recomputed or re-plotted later without re-running training."""
    os.makedirs(step_ckpt_path, exist_ok=True)
    with open(os.path.join(step_ckpt_path, "polytope_diag.json"), "w") as f:
        json.dump(stats, f, indent=2)
