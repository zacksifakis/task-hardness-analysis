# Task-Hardness Analysis — Week 01 (AIME)

**TL;DR.** Cleaned and standardized AIME generations, enforced a minimal quality gate  
(positive logprob stats + non-empty token list), and capped to 8 generations per problem  
for uniform coverage. Reasoning length (tokens) shows a clear relationship with correctness  
and will be our first baseline feature.

---

## Repo layout (week 01)

- `notebooks/week01_logprobs_fix.ipynb` — main week-01 notebook (QC + coverage + figures)
- `notebooks/reproduce.ipynb` — re-creates the main figures/tables from committed CSVs
- `notebooks/report.ipynb` — living report (narrative + figures)
- `reports/week01_qstrict_cap8/` — CSV outputs from the QC’d, capped run  
  - `per_record_stats.csv`  
  - `by_correctness_summary.csv`  
  - `coverage_by_cell.csv`  
  - `excluded_by_reason.csv` (optional)
- `figures/week01/` — exported figures used in the report  
  - `01_tok_len_boxplot.png`  
  - `02_mean_nlp_hist_correct.png`  
  - `03_mean_nlp_hist_incorrect.png`
- `experiments/log.csv` — run bookkeeping (status set to “completed” for this week’s jobs)

---

## How to reproduce the figures

**A) From committed CSVs (no re-scoring)**
1. Open `notebooks/reproduce.ipynb` in VS Code / Jupyter.
2. Run all cells. Figures go to `figures/week01/`.

**B) Rebuild the CSV report from raw outputs (if you reran generations)**
```bash
python3 nb.py \
  --bases experiment_archive results \
  --out reports/week01_qstrict_cap8 \
  --cap-per-cell 8 \
  --quality-strict
