"""Compute global polytope membership and facet-usage statistics.

Accumulate counts across chunks before computing fractions and entropy.
Averaging per-chunk statistics can bias these metrics when chunks differ in
class composition or facet assignments."""
import math

import torch


@torch.no_grad()
def global_polytope_stats(probe, acts, labels, mask, device="cpu", chunk=64):
    """Membership + facet-usage stats over the whole dataset.

    Returns frac_harmful_outside / frac_benign_outside as true global
    fractions of TOKENS, and facet entropy over the global assignment
    histogram of harmful SEQUENCES.
    """
    K = probe.num_facets
    n_h = n_h_out = n_b = n_b_out = 0
    hist = torch.zeros(K, dtype=torch.long)
    v_h_max = []
    # Count all violated facets per harmful sequence, not just the maximal facet.
    n_viol_per_seq = []
    facet_ever_violated = torch.zeros(K, dtype=torch.bool)

    for i in range(0, acts.shape[0], chunk):
        a = acts[i:i + chunk].to(device)
        m = mask[i:i + chunk].to(device)
        y = labels[i:i + chunk].to(device)

        v = probe.violations(a).float()                 # (b, S, K)
        tok = m.reshape(-1)
        v_t = v.reshape(-1, K)[tok]
        y_t = y[:, None].expand_as(m).reshape(-1)[tok]
        outside = (v_t.amax(-1) > 0)

        h = y_t > 0.5
        n_h += int(h.sum());          n_h_out += int(outside[h].sum())
        n_b += int((~h).sum());       n_b_out += int(outside[~h].sum())

        # one facet per harmful sequence, by its max-over-tokens profile
        hs = y > 0.5
        if bool(hs.any()):
            v_seq = v.masked_fill(~m[..., None], -1e4).amax(1)[hs]   # (bh, K)
            hist += torch.bincount(v_seq.argmax(-1).cpu(), minlength=K)
            v_h_max.append(v_seq.amax(-1).cpu())
            viol = (v_seq > 0)
            n_viol_per_seq.append(viol.sum(-1).float().cpu())
            facet_ever_violated |= viol.any(0).cpu()

    p = hist.float() / hist.sum().clamp(min=1)
    entropy = float(-(p * (p + 1e-10).log2()).sum())
    vh = torch.cat(v_h_max) if v_h_max else torch.zeros(1)
    nv = torch.cat(n_viol_per_seq) if n_viol_per_seq else torch.zeros(1)
    return {
        "frac_harmful_outside": n_h_out / max(n_h, 1),
        "frac_benign_outside": n_b_out / max(n_b, 1),
        "facet_entropy_bits": entropy,
        "max_entropy_bits": math.log2(K) if K > 1 else 0.0,
        "facets_argmax": int((hist > 0).sum()),
        "facets_ever_violated": int(facet_ever_violated.sum()),
        "avg_facets_violated": float(nv.mean()),
        "max_facets_violated": float(nv.max()),
        "n_facets": K,
        "harmful_v_median": float(vh.median()),
        "phi_norm_min": float(probe.phi.norm(dim=1).min()),
        "phi_norm_max": float(probe.phi.norm(dim=1).max()),
    }
