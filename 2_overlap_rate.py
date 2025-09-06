
# %%
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

# %%
# --- Directory with pre-extracted caption files (each is a list[str]) ---
HME_CAPTIONS_DIR = Path("./HME_captions")

# --- Where to persist the run log CSV ---
LOG_CSV_PATH = Path("test_overlap_log.csv")

# --- Optional: base paths ---

crohme_2014_path = Path("HME_captions/crohme_2014_captions.json")
crohme_2016_path = Path("HME_captions/crohme_2016_captions.json")
crohme_2019_path = Path("HME_captions/crohme_2019_captions.json")


crohme_train_path = Path("HME_captions/crohme_train_captions.json")
hme100k_test_path = Path("HME_captions/hme100k_test_captions.json")
hme100k_train_path = Path("HME_captions/hme100k_train_captions.json")

mne_N1_path = Path("HME_captions/N1_captions.json")
mne_N2_path = Path("HME_captions/N2_captions.json")
mne_N3_path = Path("HME_captions/N3_captions.json")

mathwriting_test_path = Path("HME_captions/mathwriting_test_captions.json")
mathwriting_valid_path = Path("HME_captions/mathwriting_valid_captions.json")
mathwriting_train_path = Path("HME_captions/mathwriting_train_captions.json")
mathwriting_synthetic_path = Path("HME_captions/mathwriting_synthetic_captions.json")

unimer_hwe = Path("HME_captions/unimer_net_hwe_captions.json")

# CROHME 2023 (not revised)
crohme_2023_train = Path("HME_captions/crohme2023_train_captions.json")
crohme_2023_test = Path("HME_captions/crohme2023_test_captions.json")
crohme_2023_val = Path("HME_captions/crohme2023_val_captions.json")

handwritten_nature_json = Path("handwritten_nature.json")
handwritten_online_json = Path("handwritten_online.json")

# =============================================================================
# UTILITIES: IO + CAPTION EXTRACTION
# =============================================================================

# %%
def _json_load(path: Path):
    """Load JSON content from path."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _is_list_of_strings(obj) -> bool:
    if isinstance(obj, list):
        obj = [str(x) for x in obj]
    else:
        raise ValueError(f"Expected list, got {type(obj)}")
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)



@lru_cache(maxsize=256)
def path2captions(path: Path) -> List[str]:
    """
    Return a list of captions for the dataset JSON.
    Auto-detects whether the JSON is:
      - a plain list[str] (already extracted captions), or
      - a list[dict] with 'messages' from which we take the last message's 'value'.
    Also writes out a mirror file under ./HME_captions/<name>_captions.json for convenience.
    """
    data = _json_load(path)
    if _is_list_of_strings(data):
        captions = list(map(str, data))
    else:
        raise ValueError(f"Unrecognized JSON format for captions: {path}")

    return captions


def load_captions_file_if_exists(path: Path) -> Optional[List[str]]:
    """
    Try to load captions from a file if it exists; return None if missing.
    """
    if not path.exists():
        return None
    return path2captions(path)

# =============================================================================
# NORMALIZATION
# =============================================================================

# %%
def normalize_caption(caption: str, *, strip_outer_math_brackets: bool = True, remove_spaces: bool = True) -> str:
    """
    Normalize a single caption :
      - strip outer \[ ... \] if present (supports raw '\\[' ... '\\]' or '[' ... ']').
      - remove all spaces.
    """
    s = str(caption)

    # Strip outer math brackets
    if strip_outer_math_brackets:
        if (s.startswith("\\[") and s.endswith("\\]")):
            s = s[2:-2]
        # if (s.startswith("[") and s.endswith("]")):
        #     s = s[1:-1]

    if remove_spaces:
        s = s.replace(" ", "")
    return s

def normalize_many(items: Iterable[str]) -> List[str]:
    """Apply normalize_caption to an iterable and return a list of normalized strings."""
    return [normalize_caption(x) for x in items]

def to_nospace_list(items: Iterable[str]) -> Set[str]:
    """Fast path to normalized set (no spaces, with bracket/prefix handling)."""
    return normalize_many(items)

def to_nospace_set(items: Iterable[str]) -> Set[str]:
    """Fast path to normalized set (no spaces, with bracket/prefix handling)."""
    return set(normalize_many(items))

# =============================================================================
# TEXTELLER SOURCES
# =============================================================================

# %%
def load_texteller_union(*paths: Path) -> Tuple[List[str], Set[str]]:
    """
    Load and combine captions from multiple Texteller sources (each is a JSON list[str]).
    Returns (raw_list, normalized_no_space_set).
    """
    combined: List[str] = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] Texteller source not found: {p}")
            continue
        data = _json_load(p)
        if not _is_list_of_strings(data):
            raise ValueError(f"Expected list[str] in {p}")
        combined.extend(data)
        print("Loaded", len(data), "items from", p)
    return combined, to_nospace_set(combined)

# =============================================================================
# LOGGING & SUMMARY
# =============================================================================

# %%
TEST_OVERLAP_LOG: List[dict] = []  # in-memory run records

def _append_to_csv(row: dict, csv_path: Path = LOG_CSV_PATH):
    """Append a single record to CSV (create with header if it doesn't exist)."""
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)

def register_overlap_result(
    *,
    test_name: str,
    test_file_path: str,
    total: int,
    overlaps: int,
    extra: Optional[dict] = None,
    persist: bool = True,
    csv_path: Path = LOG_CSV_PATH,
) -> None:
    """
    Save one run's stats to memory and optionally to CSV.
    """
    row = {
        "test_name": test_name,
        "file": str(test_file_path),
        "total": int(total),
        "overlaps": int(overlaps),
        "overlap_rate": round(overlaps / total * 100, 2) if total else 0.0,
    }
    if extra:
        row.update(extra)

    TEST_OVERLAP_LOG.append(row)
    if persist:
        _append_to_csv(row, csv_path)

def summarize_overlaps(source: str = "both", csv_path: Path = LOG_CSV_PATH) -> pd.DataFrame:
    """
    Build a summary table (DataFrame):
      - source: 'memory' | 'csv' | 'both'
    De-duplicates by (test_name, file) and sorts by time/test name.
    """
    frames: List[pd.DataFrame] = []
    if source in ("memory", "both") and TEST_OVERLAP_LOG:
        frames.append(pd.DataFrame(TEST_OVERLAP_LOG))
    if source in ("csv", "both") and csv_path.exists():
        frames.append(pd.read_csv(csv_path))

    if not frames:
        return pd.DataFrame(columns=[ "test_name", "file", "total", "overlaps", "overlap_rate"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=[ "test_name", "file"]
    )
    df = df.sort_values([ "test_name"]).reset_index(drop=True)
    return df


# =============================================================================
# CORE: OVERLAP CHECK
# =============================================================================

# %%
def check_test_overlap(
    test_file_path: Path,
    reference_captions: Iterable[str],  # raw or already normalized; we will normalize again (idempotent)
    test_name: str,
    *,
    collect: bool = True,
    persist: bool = True,
    verbose: bool = True,
    csv_path: Path = LOG_CSV_PATH,
    extra: Optional[dict] = None,
) -> Tuple[int, int]:
    """
    Compute overlap between a test dataset (captions from file) and a reference list of captions.
    - 'reference_captions' can be raw or normalized; we normalize to no-space + optional bracket/prefix handling.
    - Records to in-memory/CSV logs if desired.
    Returns (total, overlaps).
    """
    if not test_file_path.exists():
        if verbose:
            print(f"[SKIP] File not found for '{test_name}': {test_file_path}")
        return 0, 0

    test_captions = path2captions(test_file_path)
    test_captions_nospace = to_nospace_list(test_captions)  # set for fast membership

    ref_nospace_set = to_nospace_set(reference_captions)

    overlaps = sum(1 for c in test_captions_nospace if c in ref_nospace_set)
    # total = len(test_captions_nospace)
    total = len(test_captions)  # count raw entries, not unique normalized

    if verbose:
        rate = (overlaps / total * 100) if total else 0.0
        print(f"Test Name: {test_name}")
        print(f"Total: {total}, Overlaps: {overlaps} ({rate:.2f}%)")

    if collect:
        register_overlap_result(
            test_name=test_name,
            test_file_path=str(test_file_path),
            total=total,
            overlaps=overlaps,
            extra=extra,
            persist=persist,
            csv_path=csv_path,
        )

    return total, overlaps


# =============================================================================
# TEXTELLER LOADING 
# =============================================================================

# %%
def load_texteller_sources() -> Tuple[List[str], Set[str]]:
    """
    Load Texteller sources 'handwritten_online.json' + 'handwritten_nature.json' (if present).
    Returns (raw_combined_list, normalized_no_space_set).
    """
    raw_list, nospace = load_texteller_union(handwritten_online_json, handwritten_nature_json)
    print(f"Loaded Texteller sources from: {[str(handwritten_online_json), str(handwritten_nature_json)]}")
    # Print a small preview (safe even if empty)
    preview = [normalize_caption(x) for x in raw_list[:10]]
    print(f"Loaded Texteller items: {len(raw_list)}, unique normalized: {len(nospace)}")
    print(f"Preview (normalized, up to 10): {preview}")
    return raw_list, nospace

# =============================================================================
# BATCH RUNNERS
# =============================================================================

# %%

def run_batch_against_mathwriting_train():
    """
    checks using Mathwriting Train captions as the reference.
    """
    mw_train_caps = load_captions_file_if_exists(mathwriting_train_path)
    if not mw_train_caps:
        raise FileNotFoundError("Mathwriting train captions not found. Update paths in CONFIG.")

    ref = mw_train_caps  # raw list; normalization happens inside check_test_overlap

    # Your original calls
    TAG = "MathwritingTrain"  # table tag

    check_test_overlap(hme100k_test_path, ref, "HME100K Test",       extra={"CompareWith": TAG})
    check_test_overlap(hme100k_train_path, ref, "HME100K Train",     extra={"CompareWith": TAG})
    check_test_overlap(crohme_2014_path,    ref, "Crohme 2014",      extra={"CompareWith": TAG})
    check_test_overlap(crohme_2016_path,    ref, "Crohme 2016",      extra={"CompareWith": TAG})
    check_test_overlap(crohme_2019_path,    ref, "Crohme 2019",      extra={"CompareWith": TAG})
    check_test_overlap(crohme_train_path,   ref, "Crohme Train",     extra={"CompareWith": TAG})

def run_batch_against_texteller():
    """
    checks using Texteller (online + nature) as reference.
    """
    raw_texteller, _texteller_nospace = load_texteller_sources()
    ref = raw_texteller  # raw list; normalization happens in check_test_overlap
    # ref = _texteller_nospace
    TAG = "Texteller"  # table tag


    # CROHME splits
    check_test_overlap(crohme_2014_path,    ref, "Crohme 2014",      extra={"CompareWith": TAG})
    check_test_overlap(crohme_2016_path,    ref, "Crohme 2016",      extra={"CompareWith": TAG})
    check_test_overlap(crohme_2019_path,    ref, "Crohme 2019",      extra={"CompareWith": TAG})
    check_test_overlap(crohme_train_path,   ref, "Crohme Train",     extra={"CompareWith": TAG})
    # CROHME 2023
    check_test_overlap(crohme_2023_val,  ref, "Crohme 2023 Validation", extra={"CompareWith": TAG})
    check_test_overlap(crohme_2023_test, ref, "Crohme 2023 Test",       extra={"CompareWith": TAG})
    check_test_overlap(crohme_2023_train,ref, "Crohme 2023 Train",      extra={"CompareWith": TAG})
    # HME100K
    check_test_overlap(hme100k_test_path,   ref, "HME100K Test",     extra={"CompareWith": TAG})
    check_test_overlap(hme100k_train_path,  ref, "HME100K Train",    extra={"CompareWith": TAG})
    # Mathwriting
    check_test_overlap(mathwriting_test_path,        ref, "Mathwriting Test",        extra={"CompareWith": TAG})
    check_test_overlap(mathwriting_valid_path,       ref, "Mathwriting Valid",       extra={"CompareWith": TAG})
    check_test_overlap(mathwriting_train_path,ref, "Mathwriting Train", extra={"CompareWith": TAG})
    check_test_overlap(mathwriting_synthetic_path,   ref, "Mathwriting Synthetic",   extra={"CompareWith": TAG})
    # UNIMER
    check_test_overlap(unimer_hwe,           ref, "Unimer HWE",      extra={"CompareWith": TAG})

    # MNE
    check_test_overlap(mne_N1_path,         ref, "MNE N1",           extra={"CompareWith": TAG})
    check_test_overlap(mne_N2_path,         ref, "MNE N2",           extra={"CompareWith": TAG})
    check_test_overlap(mne_N3_path,         ref, "MNE N3",           extra={"CompareWith": TAG})



# =============================================================================
# MAIN (optional batch execution)
# =============================================================================

# %%
if __name__ == "__main__":
    print("=== HME Overlap Toolkit ===")
    print(f"HME_CAPTIONS_DIR: {HME_CAPTIONS_DIR.resolve()}")
    print(f"LOG_CSV_PATH: {LOG_CSV_PATH.resolve()}")
    RUN_DEFAULT_BATCH = True
    if RUN_DEFAULT_BATCH:
        # Choose which batch to run by enabling one or more of these calls.

        print("\n[Running batch against Texteller]")
        run_batch_against_texteller()
        
        print("\n[Running batch against Mathwriting Train]")
        run_batch_against_mathwriting_train()


    # Print a compact summary (memory only, then CSV if present)
    df_summary = summarize_overlaps(source="both", csv_path=LOG_CSV_PATH)
    # if not df_summary.empty:
    #     print("\n=== Summary (head) ===")
    #     print(df_summary.head(20).to_string(index=False))
    # else:
    #     print("\n[No results to summarize yet]")
