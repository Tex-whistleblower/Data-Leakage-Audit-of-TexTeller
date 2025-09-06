# TexTeller Data-Leakage Audit & PosFormer Re-evaluation

*(Code & paper artifacts for reproducible analysis)*

> **TL;DR** — This repo contains ready-to-run scripts to **quantify train–test leakage** between the *TexTeller* handwritten training corpus and public HMER benchmarks, plus a **standardized PosFormer re-evaluation pipeline** on “nano-banana” style images. Under conservative LaTeX normalization and exact matching, we find **extreme overlaps** (e.g., HME100K *test* **100%**, CROHME‑2023 *test* **94.22%**), consistent with the evidence and tables in the accompanying paper (see **Figure 1** p.1 and **Table 1** p.4).&#x20;

---

## Table of Contents

- [TexTeller Data-Leakage Audit \& PosFormer Re-evaluation](#texteller-data-leakage-audit--posformer-re-evaluation)
  - [Table of Contents](#table-of-contents)
  - [What’s in this repo](#whats-in-this-repo)
  - [Environment](#environment)
  - [Data Preparation](#data-preparation)
  - [Reproduce the Results](#reproduce-the-results)
    - [A. Overlap audit (train–test leakage)](#a-overlap-audit-traintest-leakage)
    - [B. PosFormer re-evaluation on “nano-banana” images](#b-posformer-re-evaluation-on-nano-banana-images)
  - [Key Results (match the paper)](#key-results-match-the-paper)
  - [How the matching works (conservative canonicalization)](#how-the-matching-works-conservative-canonicalization)
  - [Extend / customize](#extend--customize)
  - [FAQ](#faq)
  - [Cite](#cite)
  - [License](#license)

---

## What’s in this repo

```
.
├── 1_extract_texteller_datasets.py   # Pull TexTeller handwritten subsets from HF; export caption lists
├── 2_overlap_rate.py                 # Main script: computes overlap rates (logs to CSV, provides summary)
├── 3_reproduce_posformer.ipynb       # Standardized PosFormer reproduction (binarization + resizing + inference)
├── HME_captions/                     # Canonicalized LaTeX lists (JSON arrays) for public benchmarks
├── Texteller_en_output/              # Our reproduction of TexTeller\_en results across benchmarks
├── requirements.txt                  # Python dependencies (see Environment section)
├── nano-banana-images/               # Example “handwritten-style” images for reproduction
├── PosFormer/                        # PosFormer code bundle (configs, scripts, logs)
├── test_overlap_log.csv              # Our run log with overlap numbers (as cited in the paper)
└── README.md
```

The PDF (*Comment on arxiv 2508.09220: Data Leakage Audit of TexTeller on Public Benchmarks*) summarizes the evidence with figures/tables: **Figure 1** (p.1), **Table 1** (p.4), **Table 2** baseline (p.5), **Figure 2** visual duplicates (p.6), and PosFormer discussion (**Figure 5** p.10; **Figure 6** p.11).&#x20;

---

## Environment

* **Python**: 3.7 recommend 
* **PyTorch**: GPU optional but recommended
* **Core deps**: `datasets pandas tqdm pyyaml numpy pillow matplotlib torch torchvision pytorch-lightning`

Quick setup:

```bash
# 1) Create & activate a clean env
conda create -n hme-audit python7 -y
conda activate hme-audit

# 2) Install dependencies
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -r ./requirements.txt

```

---

## Data Preparation

1. **Export TexTeller handwritten training subsets (captions)**
   Run the script to download from Hugging Face `OleehyO/latex-formulas-80M` and export two JSON files:

```bash
python 1_extract_texteller_datasets.py
```

This will create:

* `handwritten_online.json`
* `handwritten_nature.json`

Both are **plain JSON arrays of strings** (each entry = one LaTeX expression). These two files form the **reference training corpus** in the overlap audit.

2. **Benchmark captions**
   The `HME_captions/` folder already contains JSON arrays for CROHME (2014/2016/2019), CROHME‑2023 (train/val/test), HME100K (train/test), MathWriting (train/val/test + synthetic), MNE (N1/N2/N3), and UniMER‑HWE. These are used as **evaluation splits**.

---

## Reproduce the Results

### A. Overlap audit (train–test leakage)

```bash
python 2_overlap_rate.py
```

What this does:

* Loads `handwritten_online.json` + `handwritten_nature.json` (if present) as the **TexTeller handwritten train** reference.
* Compares each `HME_captions/*.json` split **after LaTeX canonicalization** (see [How the matching works](#how-the-matching-works-conservative-canonicalization)).
* Also runs a **baseline** comparison using **MathWriting‑Train** as the reference, to show the expected *low* natural overlaps.
* Appends results to `test_overlap_log.csv` and supports an in-script summary (`summarize_overlaps()`).

The approach mirrors the quantitative protocol described in the paper (**Section 2.1–2.2**, **Table 1** on p.4; **Table 2** baseline on p.5).&#x20;

### B. PosFormer re-evaluation on “nano-banana” images

Open `3_reproduce_posformer.ipynb` (or run the cells in a Python environment) to reproduce a **standardized** PosFormer pipeline:

* **Preprocessing (critical)**

  * Convert to grayscale and **invert** (`ImageOps.invert`)
  * **Resize/limit** using `ScaleToLimitRange` to fit `H_LO/H_HI/W_LO/W_HI`
  * Create `img_mask` and pass through the model
* **Model**

  * Loads a PosFormer Lightning checkpoint, e.g.
    `PosFormer/lightning_logs/version_0/checkpoints/best.ckpt`
  * Decodes with `approximate_joint_search` and `vocab.indices2words(...)`

This adheres to **binarized (0/1) bitmap** inputs expected by standard HMER pipelines and avoids the RGB-photo mismatch noted in the paper’s critique of the original comparison (**Figure 5** p.10; **Figure 6** p.11).&#x20;

> If you see OOM or missing checkpoint errors, lower `W_HI/H_HI` or point `ckpt_path` to your local model.

---

## Key Results (match the paper)

Numbers below come from `test_overlap_log.csv` and correspond to **Table 1** (p.4) and the narrative in **Section 2.2**.&#x20;

**Overlap against TexTeller handwritten train** *(canonicalized exact matches)*

| Split                   |   Total | Found in train | Overlap (%) |
| ----------------------- | ------: | -------------: | ----------: |
| CROHME 2014 (test)      |     986 |            913 |   **92.60** |
| CROHME 2016 (test)      |   1,147 |          1,031 |   **89.89** |
| CROHME 2019 (test)      |   1,199 |          1,056 |   **88.07** |
| CROHME (train)          |   8,834 |          8,039 |       91.00 |
| CROHME‑2023 (val)       |     555 |            531 |   **95.68** |
| CROHME‑2023 (test)      |   2,300 |          2,167 |   **94.22** |
| CROHME‑2023 (train)     |  12,204 |         12,168 |   **99.71** |
| **HME100K (test)**      |  24,607 |         24,607 |  **100.00** |
| HME100K (train)         |  74,502 |         74,490 |   **99.98** |
| MathWriting (test)      |   7,644 |          6,634 |       86.79 |
| MathWriting (val)       |  15,670 |         11,744 |       74.95 |
| MathWriting (train)     | 229,836 |        166,443 |       72.42 |
| MathWriting (synthetic) | 395,711 |        228,887 |       57.84 |
| UniMER‑HWE (test)       |   6,332 |          6,000 |       94.76 |
| MNE‑N1 (test)           |   1,875 |          1,624 |       86.61 |
| MNE‑N2 (test)           |     304 |            240 |       78.95 |
| MNE‑N3 (test)           |   1,464 |            379 |       25.89 |

**Baseline** — Overlap against **MathWriting‑Train** (unrelated large set; see **Table 2** p.5).&#x20;

| Split              |  Total | Found in MW‑train | Overlap (%) |
| ------------------ | -----: | ----------------: | ----------: |
| HME100K (test)     | 24,607 |                75 |    **0.30** |
| HME100K (train)    | 74,502 |               496 |    **0.67** |
| CROHME 2014 (test) |    986 |                34 |    **3.45** |
| CROHME 2016 (test) |  1,147 |                18 |    **1.57** |
| CROHME 2019 (test) |  1,199 |                13 |    **1.08** |
| CROHME (train)     |  8,834 |               670 |    **7.58** |

> Visual spot checks (paper **Figure 2** p.6) show many matches are **identical** or **trivially augmented** images, not just label-string coincidences.&#x20;

---

## How the matching works (conservative canonicalization)

We intentionally use a **minimal** LaTeX normalization followed by **exact string equality**:

* **Strip outer display delimiters**: remove a single outer pair of `\[ ... \]` if present.
* **Remove all whitespace**: spaces/newlines are deleted everywhere.
* **Exact match**: after canonicalization, a test item “overlaps” iff its string **appears at least once** in the canonicalized training labels.

See `normalize_caption`, `to_nospace_set/list`, and `check_test_overlap` in `2_overlap_rate.py`. This is the same conservative policy described in the paper’s **Section 2.1** (LaTeX Normalization & matching protocol).&#x20;

---

## Extend / customize

* **Audit your own split**
  Save your expressions as `my_test.json` (plain string array) and add:

  ```python
  from pathlib import Path
  my_test = Path("my_test.json")
  raw_texteller, _ = load_texteller_sources()  # from 2_overlap_rate.py
  check_test_overlap(my_test, raw_texteller, "My Test", extra={"CompareWith": "Texteller"})
  ```

  Run `python 2_overlap_rate.py` to append results to `test_overlap_log.csv`.

* **Try different normalization rules**
  If you want to explore stricter or looser canonicalization (e.g., handling `\left...\right` pairs or synonym tokens), edit `normalize_caption`. Be sure to **report which rules** you used to keep results comparable.

---

## FAQ

**Q1. I don’t see `handwritten_online.json` / `handwritten_nature.json`.**
Run `python 1_extract_texteller_datasets.py` first (and ensure you can access the Hugging Face dataset).

**Q2. `FileNotFoundError` for MathWriting baseline.**
`run_batch_against_mathwriting_train()` expects `HME_captions/mathwriting_train_captions.json`. If you only need the TexTeller-vs-benchmarks audit, you can comment out that baseline batch. The default script runs **both** batches.

**Q3. Why binarize for PosFormer?**
PosFormer and most HMER pipelines assume **binarized** bitmaps, not RGB photos. Using RGB can yield misleading failures. The paper highlights this evaluation mismatch and re-runs PosFormer with a **canonical binarized pipeline** (see **Figure 5** p.10 and **Figure 6** p.11).&#x20;

**Q4. Are these overlaps “just labels” or actual image duplicates?**
Section 2.3 shows side‑by‑side images indicating many are identical or trivially augmented copies (**Figure 2** p.6).&#x20;

---

## Cite

If you use this repository or the analysis in your work, please cite the paper (fill in final bibliographic details as appropriate):

```bibtex
@misc{hmer_team_texteller_audit,
  title  = {Comment on arxiv 2508.09220: Data Leakage Audit of TexTeller on Public Benchmarks},
  author = {HMER Team},
  year   = {2025},
  note   = {See the PDF included in this repository},
}
```

Relevant figures/tables are referenced inline (e.g., **Table 1** p.4 and **Table 2** p.5).&#x20;

---

## License

Please add your preferred license terms here (e.g., MIT/Apache‑2.0) and **respect third‑party licenses** for bundled code (e.g., `PosFormer/`).

---

**Contact** — Questions or suggestions? Please open an issue.
