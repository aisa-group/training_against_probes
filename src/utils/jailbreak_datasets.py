"""Load training prompts and labeled completion datasets for probe fitting.
"""

import os
import random
from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset

HF_DATASET = os.environ.get("PROBE_COMPLETIONS_DATASET", "lenalibon/jailbreak-judge-completions")

# Which model's completions the probes are fitted on. The dataset carries one
# completion column per source model, and the probe must be fitted on the
# completions of the model it will later supervise -- fitting on the heretic
# Llama's completions and then applying the probe to Mistral activations would
# be measuring the wrong distribution.
#
# Override with PROBE_COMPLETION_MODEL, e.g. "mistral_7b_instruct_v01".
# Available: llama3_8b_abliterated, llama3_8b_heretic_mlabonne,
#            mistral_7b_instruct_v01
_COMPLETION_MODEL = os.environ.get("PROBE_COMPLETION_MODEL", "llama3_8b_heretic_mlabonne")
COMPLETION_COL = f"completion_{_COMPLETION_MODEL}"
LABEL_COL = f"response_harmfulness_{_COMPLETION_MODEL}"

# Paired probe-training set (paired-data ablation): one question per row with a
# harmful and a benign continuation, supplied as a prepared parquet file.
PAIRED_PATH = str(Path(os.environ.get("DATASETS_DIR", Path(os.environ.get("NM_ROOT", "/data/new_master")) / "datasets")) / "wildguardtest_paired_v1.parquet")


def load_split(split_name: str) -> HFDataset:
    """Load a named split and rename the completion column to 'completion'."""
    ds = load_dataset(HF_DATASET, split=split_name)
    ds = ds.rename_column(COMPLETION_COL, "completion")
    # Normalise string labels ("harmful"/"unharmful") to integers (1/0)
    if ds.features[LABEL_COL].dtype == "string":
        ds = ds.map(lambda x: {LABEL_COL: 1 if x[LABEL_COL] == "harmful" else 0})
    return ds


def empty_dataset() -> HFDataset:
    return HFDataset.from_dict({"prompt": [], "completion": []})


def _filter_label(ds: HFDataset, label: int) -> HFDataset:
    return ds.filter(lambda x: x[LABEL_COL] == label)


def _filled_negatives(primary: HFDataset, secondary: HFDataset, n: int) -> HFDataset:
    """Return up to n negatives (label=0), taken from primary first, filling remainder from secondary.

    Scans primary for label=0 rows, takes however many are found (up to n),
    then fills any remaining budget from secondary.
    """
    from_primary = _filter_label(primary, 0)
    n_primary = min(len(from_primary), n)
    remaining = n - n_primary

    parts = []
    if n_primary > 0:
        parts.append(from_primary.select(range(n_primary)))
    if remaining > 0:
        from_secondary = _filter_label(secondary, 0)
        n_secondary = min(len(from_secondary), remaining)
        if n_secondary > 0:
            parts.append(from_secondary.select(range(n_secondary)))

    return concatenate_datasets(parts) if parts else empty_dataset()


def build_probe_datasets(
    train_pairs: list[tuple[str, str]],
    eval_pairs: list[tuple[str, str]],
    n_train: int,
    n_eval: int,
) -> tuple[
    HFDataset,                               # train positives
    HFDataset,                               # train negatives
    dict[str, tuple[HFDataset, HFDataset]],  # pair_name → (eval_pos, eval_neg)
]:
    """Load splits, enforce train/eval non-overlap, and return ready-to-use datasets.

    For each harmful split used in training, the train window extends up to (and
    including) the row where the n_train-th label=1 sample appears.  Eval starts
    from the next row.  Both train and eval use the full benign split (no windowing).

    Positive/negative convention (same for train and eval):
      positives  = label=1 rows from the harmful split, up to n.
      negatives  = label=0 rows from the harmful split (model refusals) within
                   the same row window, topped up from the full benign split.

    Args:
        train_pairs: (harmful_split, benign_split) pairs for training.
        eval_pairs:  (harmful_split, benign_split) pairs for evaluation.
        n_train: Max samples per class for training (across all train pairs combined).
        n_eval:  Max samples per class per eval pair.

    Returns:
        (train_pos, train_neg, eval_datasets) where eval_datasets maps
        "{harmful}_vs_{benign}" to (eval_pos, eval_neg).
    """
    all_split_names = {s for pair in train_pairs + eval_pairs for s in pair}
    eval_split_names = {s for pair in eval_pairs for s in pair}

    # Load all splits once
    loaded: dict[str, HFDataset] = {name: load_split(name) for name in all_split_names}

    # For each harmful split used in training, find the row index (exclusive) that
    # contains the n_train-th label=1 sample.  All rows up to that index form the
    # train window; rows from that index onward are available for eval.
    # Benign splits are not windowed — they are used in full for both train and eval.
    train_harmful_splits = {pair[0] for pair in train_pairs}

    def _find_train_cutoff(name: str) -> int:
        """Return the exclusive row index after the n_train-th label=1 row."""
        count = 0
        for i, label in enumerate(loaded[name][LABEL_COL]):
            if label == 1:
                count += 1
                if count >= n_train:
                    return i + 1
        return len(loaded[name])  # fewer than n_train positives in the split

    train_cutoff: dict[str, int] = {
        name: _find_train_cutoff(name) if name in train_harmful_splits else 0
        for name in all_split_names
    }

    def _train_slice(name: str) -> HFDataset:
        """Rows [0 : cutoff] for harmful splits; empty otherwise."""
        end = train_cutoff[name]
        return loaded[name].select(range(0, end)) if end > 0 else empty_dataset()

    def _train_benign_slice(name: str) -> HFDataset:
        """Full dataset — benign splits are not windowed."""
        return loaded[name]

    def _eval_slice(name: str) -> HFDataset:
        """Rows [cutoff : end] for harmful splits that were also trained on;
        rows [0 : end] for splits only used in eval."""
        start = train_cutoff[name]
        end = len(loaded[name])
        return loaded[name].select(range(start, end)) if start < end else empty_dataset()

    def _eval_benign_slice(name: str) -> HFDataset:
        """Full dataset — benign splits are not windowed."""
        return loaded[name]

    # ----------------------------------------------------------------
    # Training data — collect across all pairs, then cap per class
    # ----------------------------------------------------------------
    train_pos_parts: list[HFDataset] = []
    train_neg_parts: list[HFDataset] = []

    for harmful_split, benign_split in train_pairs:
        harmful_train = _train_slice(harmful_split)
        benign_train = _train_benign_slice(benign_split)

        pos = _filter_label(harmful_train, 1)
        if len(pos) > 0:
            train_pos_parts.append(pos)

        neg = _filled_negatives(harmful_train, benign_train, n_train)
        if len(neg) > 0:
            train_neg_parts.append(neg)

    all_train_pos = concatenate_datasets(train_pos_parts) if train_pos_parts else empty_dataset()
    all_train_neg = concatenate_datasets(train_neg_parts) if train_neg_parts else empty_dataset()

    train_pos = all_train_pos.select(range(min(n_train, len(all_train_pos))))
    train_neg = all_train_neg.select(range(min(n_train, len(all_train_neg))))

    # ----------------------------------------------------------------
    # Eval data — one (pos, neg) tuple per pair
    # ----------------------------------------------------------------
    eval_datasets: dict[str, tuple[HFDataset, HFDataset]] = {}

    for harmful_split, benign_split in eval_pairs:
        pair_name = f"{harmful_split}_vs_{benign_split}"

        harmful_eval = _eval_slice(harmful_split)
        benign_eval = _eval_benign_slice(benign_split)

        eval_pos_all = _filter_label(harmful_eval, 1)
        eval_pos = eval_pos_all.select(range(min(n_eval, len(eval_pos_all))))
        eval_neg = _filled_negatives(harmful_eval, benign_eval, n_eval)

        eval_datasets[pair_name] = (eval_pos, eval_neg)

    return train_pos, train_neg, eval_datasets


def build_paired_probe_datasets(
    n_train: int,
    n_eval: int,
    path: str = PAIRED_PATH,
) -> tuple[
    HFDataset,                               # train positives (harmful completion)
    HFDataset,                               # train negatives (benign completion)
    dict[str, tuple[HFDataset, HFDataset]],  # {"paired_harmful_vs_benign": (eval_pos, eval_neg)}
]:
    """Load the paired probe-training set (same question, two continuations).

    Positives and negatives differ ONLY in which completion column is exposed as
    "completion" — the "prompt" column is identical between the two sides of a
    pair, so the probe cannot key on the question.

    Selection is deterministic (row order + select(range(...)), no RNG), so a
    warm-start retrain reuses byte-identical training data, matching the
    behaviour of build_probe_datasets.
    """
    ds = load_dataset("parquet", data_files=path, split="train")

    def _side(d: HFDataset, keep: str, drop: str) -> HFDataset:
        cols_to_remove = [c for c in (keep, drop, "source_harmful", "source_benign",
                                      "pair_split") if c in d.column_names and c != keep]
        d = d.remove_columns([c for c in cols_to_remove if c != "prompt"])
        return d.rename_column(keep, "completion")

    H, B = "completion_harmful", "completion_benign"
    tr = ds.filter(lambda x: x["pair_split"] == "train")
    ev = ds.filter(lambda x: x["pair_split"] == "eval")
    tr = tr.select(range(min(n_train, len(tr))))
    ev = ev.select(range(min(n_eval, len(ev))))

    train_pos = _side(tr, H, B)
    train_neg = _side(tr, B, H)
    eval_pos = _side(ev, H, B)
    eval_neg = _side(ev, B, H)

    return train_pos, train_neg, {"paired_harmful_vs_benign": (eval_pos, eval_neg)}


# ──────────────────────────── BeaverTails ────────────────────────────────────
# SaP's own fitting dataset (PKU-Alignment/BeaverTails, 330k_train). Unlike
# wildguardtest, which is jailbreak-shaped, this is general harmful/benign
# conversation across 14 harm categories -- which is what makes a per-facet
# category analysis possible at all, and what SaP used for their facet
# interpretability figures.
#
# Rows: prompt, response, category (dict of 14 booleans), is_safe.
# Label 1 denotes harmful content (is_safe=False).

BEAVERTAILS_DATASET = "PKU-Alignment/BeaverTails"
BEAVERTAILS_CATEGORIES = [
    "animal_abuse", "child_abuse", "controversial_topics,politics",
    "discrimination,stereotype,injustice", "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft", "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior", "privacy_violation", "self_harm",
    "sexually_explicit,adult_content", "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]


def _beavertails_primary_category(row) -> str:
    """Return the first flagged harm category, or 'none' for safe rows.

    Selecting one category gives each row a single label for grouped analysis.
    """
    cat = row.get("category") or {}
    for c in BEAVERTAILS_CATEGORIES:
        if cat.get(c):
            return c
    return "unlabelled_harmful" if not row.get("is_safe", True) else "none"


def build_beavertails_probe_datasets(
    n_train: int,
    n_eval: int = 0,
    split: str = "330k_train",
    eval_split: str = "330k_test",
    seed: int = 42,
    max_scan: int = 60000,
):
    """(train_pos, train_neg, {pair_name: (eval_pos, eval_neg)}) from BeaverTails.

    Same shape as build_probe_datasets, so it drops into the existing probe
    fitting path. Rows are shuffled with a fixed seed and balanced per class, and
    the primary harm category is carried through as `bt_category` for the
    per-facet analysis.
    """
    from datasets import Dataset as _HFDataset

    def _take(split_name: str, n: int):
        if n <= 0:
            return None, None
        ds = load_dataset(BEAVERTAILS_DATASET, split=split_name)
        ds = ds.shuffle(seed=seed).select(range(min(max_scan, len(ds))))
        pos, neg = [], []
        for r in ds:
            bucket = pos if not r["is_safe"] else neg
            if len(bucket) >= n:
                if len(pos) >= n and len(neg) >= n:
                    break
                continue
            bucket.append({
                "prompt": r["prompt"],
                "completion": r["response"],
                "bt_category": _beavertails_primary_category(r),
            })
        return _HFDataset.from_list(pos), _HFDataset.from_list(neg)

    train_pos, train_neg = _take(split, n_train)
    eval_sets = {}
    if n_eval > 0:
        ep, en = _take(eval_split, n_eval)
        if ep is not None:
            eval_sets["beavertails_harmful_vs_beavertails_safe"] = (ep, en)
    return train_pos, train_neg, eval_sets


def build_beavertails_paired_probe_datasets(
    n_train: int,
    n_eval: int = 0,
    split: str = "330k_train",
    eval_split: str = "330k_test",
    seed: int = 42,
):
    """PAIRED BeaverTails: the harmful and benign example share a prompt.

    build_beavertails_probe_datasets above fills the two class buckets from
    independently shuffled ROWS, so the harmful and benign examples almost never
    come from the same prompt. Measured at n_train=750: 455/750 harmful rows and
    604/750 benign rows come from prompts that do have both kinds, but only 16
    PROMPTS appear in both buckets. In 98% of cases the two classes differ in
    topic as well as in harmfulness, so a detector can score well by keying on
    what the conversation is ABOUT rather than on whether the response is
    harmful.

    This builder removes that confound. It keeps only prompts that have at least
    one safe and at least one unsafe response (10,729 of the 16,188 unique
    prompts in 330k_train, 66.3%), and emits one of each per prompt. The result
    is the same 2*n_train rows but with prompt content controlled, and it is
    exactly the set a DPO run can consume as (prompt, chosen=safe,
    rejected=unsafe).

    Selection effect, stated because it is real: dropping the 22.7% of prompts
    with only unsafe responses removes the ones the annotated model never
    answered safely, which are plausibly the hardest. A detector fitted here is
    fitted on an easier prompt distribution than the unpaired builder uses.

    Returns the same (train_pos, train_neg, eval_sets) shape as its unpaired
    sibling, so it drops into the existing fitting path unchanged. Row i of
    train_pos and row i of train_neg share a prompt.
    """
    from collections import defaultdict

    from datasets import Dataset as _HFDataset

    def _take(split_name: str, n: int):
        if n <= 0:
            return None, None
        ds = load_dataset(BEAVERTAILS_DATASET, split=split_name)
        by = defaultdict(lambda: ([], []))          # prompt -> (unsafe, safe)
        for r in ds:
            by[r["prompt"]][0 if not r["is_safe"] else 1].append(r)
        both = [p for p, (u, s) in by.items() if u and s]
        rng = random.Random(seed)
        rng.shuffle(both)
        if len(both) < n:
            raise ValueError(
                f"{split_name}: only {len(both)} prompts have both a safe and an "
                f"unsafe response, need {n}")
        pos, neg = [], []
        for p in both[:n]:
            u, s = by[p]
            # One response per class per prompt, chosen with the same seeded rng
            # so the pairing is reproducible rather than order-dependent.
            ru, rs = rng.choice(u), rng.choice(s)
            pos.append({"prompt": p, "completion": ru["response"],
                        "bt_category": _beavertails_primary_category(ru)})
            neg.append({"prompt": p, "completion": rs["response"],
                        "bt_category": _beavertails_primary_category(rs)})
        return _HFDataset.from_list(pos), _HFDataset.from_list(neg)

    train_pos, train_neg = _take(split, n_train)
    eval_sets = {}
    if n_eval > 0:
        ep, en = _take(eval_split, n_eval)
        if ep is not None:
            eval_sets["beavertails_harmful_vs_beavertails_safe"] = (ep, en)
    return train_pos, train_neg, eval_sets


def build_harmbench_probe_datasets(
    n_train: int,
    n_eval: int = 0,
    path: str = "/data/new_master/datasets/harmbench_mistral.parquet",
    seed: int = 42,
):
    """(train_pos, train_neg, eval_sets) from the HarmBench fitting parquet.

    Requires a prepared parquet containing HarmBench prompts
    plus real GCG suffixes, the target model's own completions, StrongREJECT
    labels. This is SaP's stronger setting -- GCG is the top-ASR attack they fit
    on (Llama-2: GCG 30, AutoPrompt 16.5, UAT 6.5, excluding the eval-only
    AdaptiveAttack).

    Same shape as build_probe_datasets so it drops into the existing fit path.
    Train/eval are split disjointly so evaluation never sees fitted rows.
    """
    from datasets import Dataset as _HFDataset

    ds = load_dataset("parquet", data_files=path, split="train").shuffle(seed=seed)
    pos = [r for r in ds if r["label"] == 1]
    neg = [r for r in ds if r["label"] == 0]

    def _cut(rows, a, b):
        return _HFDataset.from_list([
            {"prompt": r["prompt"], "completion": r["completion"],
             "bt_category": r.get("category", "unknown")}
            for r in rows[a:b]
        ])

    tr_pos, tr_neg = _cut(pos, 0, n_train), _cut(neg, 0, n_train)
    eval_sets = {}
    if n_eval > 0:
        ep, en = _cut(pos, n_train, n_train + n_eval), _cut(neg, n_train, n_train + n_eval)
        if len(ep) and len(en):
            eval_sets["harmbench_harmful_vs_harmbench_benign"] = (ep, en)
    return tr_pos, tr_neg, eval_sets


# ──────────────────────────── Liar's Bench (deception) ───────────────────────
# Liar's Bench instructed-deception rows contain system, user, and assistant turns.
# Preserve the system instruction when constructing prompts. Label 1 denotes
# deception and label 0 denotes honesty. Split the available test data into
# disjoint fitting and evaluation subsets with a fixed seed.

LIARS_BENCH_DATASET = "Cadenza-Labs/liars-bench"


def _lb_roles(messages) -> tuple:
    """Extract system, user, and assistant contents by role from a dataset row.
    """
    sys_c = usr_c = asst_c = None
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content")
        if role == "system" and sys_c is None:
            sys_c = content
        elif role in ("user", "human") and usr_c is None:
            usr_c = content
        elif role == "assistant" and asst_c is None:
            asst_c = content
    return sys_c, usr_c, asst_c


def _lb_prompt(sys_c, usr_c) -> str:
    """Combine system instructions and user text into one prompt.

    The training tokenizer accepts a single user turn. Including the system text
    preserves the instruction to answer honestly or deceptively during both probe
    fitting and on-policy generation.
    """
    parts = [p for p in (sys_c, usr_c) if p]
    return "\n\n".join(parts)


def _lb_parquet_path(config: str, split: str) -> str:
    """Locate the cached Liar's Bench parquet for a configuration and split.

    Reading parquet directly avoids resolving the List feature type, which is
    unsupported by the datasets version in the pinned training environment.
    """
    import glob as _glob
    import os as _os
    roots = [
        _os.environ.get("HF_HOME", ""),
        _os.path.expanduser("~/.cache/huggingface"),
    ]
    for root in roots:
        if not root:
            continue
        hits = sorted(_glob.glob(_os.path.join(
            root, "hub", "datasets--Cadenza-Labs--liars-bench", "snapshots", "*",
            config, f"{split}-*.parquet")))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"liars-bench {config}/{split} parquet not found in HF cache (roots={roots})")


def _lb_load_rows(config: str, split: str, seed: int) -> list:
    """Shuffled list of row-dicts (messages, deceptive, dataset) from the parquet."""
    import random as _random
    import pyarrow.parquet as _pq
    rows = _pq.read_table(_lb_parquet_path(config, split)).to_pylist()
    _random.Random(seed).shuffle(rows)
    return rows


def build_liarsbench_probe_datasets(
    n_train: int,
    n_eval: int = 0,
    config: str = "instructed-deception",
    split: str = "test",
    seed: int = 42,
):
    """(train_pos, train_neg, eval_sets) from Cadenza-Labs/liars-bench.

    Same shape/return contract as build_beavertails_probe_datasets so it drops
    straight into _build_probe_fitting_data. Positives are deceptive rows
    (label 1), negatives honest rows (label 0). prompt = folded system+user,
    completion = the assistant answer. Selection is deterministic (fixed-seed
    shuffle + disjoint index ranges), so a warm-start refit reuses byte-identical
    fitting rows and train/eval never overlap.
    """
    from datasets import Dataset as _HFDataset

    ds = _lb_load_rows(config, split, seed)

    dec: list[dict] = []
    hon: list[dict] = []
    for r in ds:
        sys_c, usr_c, asst_c = _lb_roles(r["messages"])
        if not usr_c or not asst_c:
            continue
        row = {
            "prompt": _lb_prompt(sys_c, usr_c),
            "completion": asst_c,
            # Carried through for a per-source-dataset analysis, mirroring
            # bt_category in the BeaverTails builders.
            "lb_dataset": r.get("dataset", "unknown"),
        }
        (dec if r["deceptive"] else hon).append(row)

    def _cut(rows: list[dict], a: int, b: int):
        return _HFDataset.from_list(rows[a:b])

    train_pos = _cut(dec, 0, n_train)
    train_neg = _cut(hon, 0, n_train)

    eval_sets: dict = {}
    if n_eval > 0:
        eval_pos = _cut(dec, n_train, n_train + n_eval)
        eval_neg = _cut(hon, n_train, n_train + n_eval)
        if len(eval_pos) and len(eval_neg):
            eval_sets["liars_deceptive_vs_honest"] = (eval_pos, eval_neg)

    return train_pos, train_neg, eval_sets


def load_liarsbench_prompts(
    config: str = "instructed-deception",
    side: str = "deceptive",
    split: str = "test",
    max_n: "int | None" = None,
    seed: int = 42,
) -> list[str]:
    """On-policy prompt pool for the deception finetuning loop.

    Returns folded system+user prompt strings for one side of Liar's Bench:
      side="deceptive" -> rows whose system turn instructs deception
      side="honest"    -> rows whose system turn instructs truthfulness

    The model generates completions for these prompts during finetuning and the
    deception probe scores those completions; the "deceptive" pool plays the role
    the harmful pool plays in the harmfulness runs, "honest" the benign role.

    Prompts are DEDUPED: liars-bench repeats each (system, user) pair once per
    source model (5x here), and on-policy generation should not sample the same
    prompt five times as often.
    """
    if side not in ("deceptive", "honest"):
        raise ValueError(f"side must be 'deceptive' or 'honest', got {side!r}")
    want_deceptive = side == "deceptive"

    ds = _lb_load_rows(config, split, seed)
    prompts: list[str] = []
    seen: set = set()
    for r in ds:
        if bool(r["deceptive"]) != want_deceptive:
            continue
        sys_c, usr_c, _ = _lb_roles(r["messages"])
        if not usr_c:
            continue
        p = _lb_prompt(sys_c, usr_c)
        if p in seen:
            continue
        seen.add(p)
        prompts.append(p)
        if max_n is not None and len(prompts) >= max_n:
            break
    if not prompts:
        raise ValueError(
            f"no {side} prompts found in {LIARS_BENCH_DATASET}:{config}:{split}")
    return prompts


# ──────────────────────────── JailbreakBench prompts ─────────────────────────

JBB_DS = "JailbreakBench/JBB-Behaviors"
JBB_N = 100          # the harmful split is exactly 100 behaviours


def load_jailbreakbench() -> list[str]:
    """The 100 JailbreakBench harmful behaviours used for StrongREJECT.

    This is the prompt set behind every StrongREJECT number in the figures. The
    column name has moved between dataset revisions, hence the fallback list.
    """
    ds = load_dataset(JBB_DS, "behaviors", split="harmful")
    for col in ("Goal", "goal", "Behavior", "behavior", "prompt"):
        if col in ds.column_names:
            return [ex[col] for ex in ds][:JBB_N]
    str_cols = [c for c, f in ds.features.items() if str(f.dtype) == "string"]
    if str_cols:
        return [ex[str_cols[0]] for ex in ds][:JBB_N]
    raise ValueError(f"No usable text column found in {JBB_DS}. "
                     f"Columns: {ds.column_names}")
