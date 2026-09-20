# Do Conformal Guarantees Expire? Label-Free Recalibration of Prediction Sets on a Decade-Long Stream of Scientific Abstracts

Code, results, and paper source for the IEEE BigData 2026 High School Symposium submission by Jeremiah Joseph (Apex High School). Everything was produced by running the experiments in `code/`;
machine (no numbers were invented). All numbers in the paper come from
`results/final_results.json`, printed in readable form in `results/summary_tables.txt`.

## What's here

| Path | What it is |
|---|---|
| `Joseph_IEEE_BigData_2026_HS_Symposium.pdf` | The submission PDF (5 pages, IEEEtran conference format). |
| `paper/main.tex`, `paper/refs.bib`, `paper/fig*.pdf` | LaTeX source. Upload the whole `paper/` folder to Overleaf to edit. |
| `code/extract.py` | Pulls all cs.* papers with dates from the arXiv metadata snapshot (HF `librarian-bots/arxiv-metadata-snapshot`). |
| `code/sample_embed.py` | Samples 600 papers/month (2013-01 → 2026-08, 12 categories) and embeds them with MiniLM. ~32 min on CPU. |
| `code/final.py` | All conformal experiments (static, Mondrian, sliding, weighted, ACI, periodic, random, SelfCal, SelfTrig). ~2 min. |
| `code/figs.py` | Makes the three figures. |
| `code/table.py` | Prints the summary tables. |
| `code/analyze_stats.py` | Correlation study of label-free statistics vs true coverage (Sec. V-B). |
| `data/` (not in repo, 212 MB) | Regenerate with `extract.py` + `sample_embed.py` (~40 min on CPU). |

## Reproduce

```bash
python3 -m venv venv && ./venv/bin/pip install -r code/requirements.txt
cd code
# (optional, ~40 min) rebuild data from scratch:
#   python extract.py && python sample_embed.py
python final.py     # writes final_results.json
python figs.py      # writes ../paper/fig*.pdf
python table.py     # prints the tables
```

`final.py` expects `data/sample.parquet` and `data/emb_minilm.npy` relative to the
working directory — run it from the folder that contains `data/` (or symlink).

## Submission rules (from the symposium page)
- Max **5 pages including references** (we are at exactly 5).
- IEEE conference template (IEEEtran, used).
- First author must be a high-school student at time of submission; single-blind review.
- In-person attendance required if accepted (Dec 17, 2026).
