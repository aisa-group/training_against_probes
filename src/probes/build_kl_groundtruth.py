"""Prepare KL reference text from dataset answers.

Requires a datasets version compatible with the selected source; no GPU is needed.
Writes {ROOT}/datasets/{source}_kl_groundtruth.json as a list of prompt/completion
pairs, compatible with utils.finetune_utils.load_kl_rollouts."""
import json
import os
import random
import sys

ROOT = os.environ.get("NM_ROOT", "/data/new_master")
OUT = os.path.join(ROOT, "datasets")
N = int(os.environ.get("N_PROMPTS", "5000"))
SEED = 42

SUBJECTS = ["algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus"]


def pairs_math():
    """(problem, the reference worked solution). MATH ships a full solution."""
    from datasets import load_dataset
    rows = []
    for subj in SUBJECTS:
        for ex in load_dataset("EleutherAI/hendrycks_math", subj, split="train"):
            if ex.get("problem") and ex.get("solution"):
                rows.append((ex["problem"], ex["solution"]))
    return rows


def _first_exchange(msgs):
    """The opening (user, assistant) turn of a chat record, or None."""
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    u, a = msgs[0], msgs[1]
    if not (isinstance(u, dict) and isinstance(a, dict)):
        return None
    ur = (u.get("role") or u.get("from") or "").lower()
    ar = (a.get("role") or a.get("from") or "").lower()
    uc = u.get("content") or u.get("value")
    ac = a.get("content") or a.get("value")
    if uc and ac and ur in ("user", "human") and ar in ("assistant", "gpt", "model"):
        return (uc, ac)
    return None


def pairs_chat(name, split):
    """(user turn, the dataset's assistant reply) for an instruction set.

    The existing prompt loader keeps only the user turn and discards the reply,
    which is exactly the half needed here, so this cannot reuse it.
    """
    from datasets import load_dataset
    rows = []
    for ex in load_dataset(name, split=split):
        p = _first_exchange(ex.get("messages") or ex.get("conversations")
                            or ex.get("conversation"))
        if p:
            rows.append(p)
        elif ex.get("prompt") and (ex.get("response") or ex.get("completion")):
            rows.append((ex["prompt"], ex.get("response") or ex["completion"]))
    return rows


SOURCES = {
    "math": pairs_math,
    "ultrachat": lambda: pairs_chat("HuggingFaceH4/ultrachat_200k", "train_sft"),
    "dolci": lambda: pairs_chat("allenai/Dolci-Instruct-SFT", "train"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    want = sys.argv[1:] or list(SOURCES)
    for src in want:
        if src not in SOURCES:
            raise SystemExit(f"unknown source {src!r}; pick from {list(SOURCES)}")
        rows = SOURCES[src]()
        if not rows:
            raise SystemExit(f"{src}: no (prompt, reference) pairs recovered")
        random.Random(SEED).shuffle(rows)
        rows = rows[:N]
        out = [{"prompt": p, "completion": c} for p, c in rows]
        path = os.path.join(OUT, f"{src}_kl_groundtruth.json")
        json.dump(out, open(path, "w"))
        lens = [len(c.split()) for _, c in rows]
        print(f"[gt] {src}: {len(out)} pairs, reference length mean "
              f"{sum(lens) / len(lens):.0f} words (min {min(lens)}, max {max(lens)})")
        print(f"[gt] wrote {path}")
        print(f"[gt]   sample reference: {rows[0][1][:160]!r}")


if __name__ == "__main__":
    main()
