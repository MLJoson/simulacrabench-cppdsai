# SimulacraBench 2026 — Team CPP DSAI

Submission repository for **Team CPP DSAI** (Cal Poly Pomona Data Science & Artificial Intelligence Club) in [SimulacraBench](https://simulacrabench.org), a NeurIPS 2026 competition on conditional survey response imputation. Given observed respondent features (`GIVEN`) and schema metadata, our pipeline predicts calibrated categorical probability distributions over held-out survey items (`PREDICT`) across the **UNICEF**, **World Bank**, and **UNHCR** benchmarks.

Full technical details and ablation notes are available in [`main.pdf`](main.pdf).

---

## Leaderboard Results

| Split | `skill` (Mean) | `skill_unicef` | `skill_world_bank` | `skill_unhcr` |
| :--- | :---: | :---: | :---: | :---: |
| **Development** | 0.386 | 0.350 | 0.234 | 0.576 |
| **Test** (Submission `1061318`, 2026-11-19) | **0.300** | **0.340** | **0.230** | **0.330** |

---




## Repository Guide

| File | Description |
| :--- | :--- |
| `main.py` | End-to-end submission entrypoint (`predict`), schema-adaptive model routing, `SurveyMLP`, skip-logic gating, and calibration. |
| `requirements.txt` | Pinned Python package versions used for the official Codabench submission[cite: 2]. |
| `main.pdf` | NeurIPS 2026 SimulacraBench technical report. |
| `LICENSE` | MIT License file. |

## Reproduction

Clone the repository and install the pinned dependencies (Python 3.12+ with a CUDA-enabled GPU recommended for CatBoost and PyTorch):

```bash
git clone [https://github.com/MLJoson/simulacrabench-cpp-dsai.git](https://github.com/MLJoson/simulacrabench-cpp-dsai.git)
cd simulacrabench-cpp-dsai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Data Use & Compliance

At inference time, `predict(frame, schema)` operates strictly on the input DataFrame and schema metadata provided by the evaluation harness, with no external data or pretrained weights. No respondent microdata is stored or redistributed in this repository, and no attempt was made to re-identify respondents or link records to external sources.