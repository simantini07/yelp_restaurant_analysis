# 🍽️ Topic, Trend & Reviewer Graph: A Multi-Lens Closure Risk Model for Philadelphia Restaurants

> **Why do some restaurants close even when their star ratings still look fine?**

Stars are a single noisy number — they collapse food quality, service, price, and trend into one digit. This project asks whether **review text, sentiment trends over time, and reviewer-network behaviour** can do a better job of predicting restaurant closure than the star average alone. Using **5,856 Philadelphia restaurants** and **300,000 Yelp reviews**, we layer three independent evidence streams — *what* people write, *how* their sentiment moves, and *who* writes the reviews and combine them into a single closure-risk model that takes prediction from baseline AUC ≈ 0.72 to **AUC ≈ 0.94**, while keeping every signal interpretable.

🎥 **Project video:** <https://www.youtube.com/watch?v=V9y9EpvLQzc>

👉 **Main deliverable:** [`main_notebook.ipynb`](./main_notebook.ipynb)

---

## 🔍 Research Questions

| # | Question | Signal | Methods |
|---|----------|--------|---------|
| **RQ1** | Does *what* customers write predict closure beyond stars? | Review text | TF–IDF + Logistic Regression, **LDA topics**, **BERTopic-style neural embeddings**, hybrid `HistGradientBoostingClassifier` |
| **RQ2** | Are closing restaurants quietly trending downward over time? | Sentiment trajectory | **VADER** sentiment → quarterly series → **Mann–Kendall** trend test + Mann–Whitney U on slopes |
| **RQ3** | Does it matter *who* the reviewers are? | Reviewer network + behaviour | Bipartite reviewer↔business graph + **PageRank**, **Isolation Forest** - enriched 15-feature reviewer profiles |

---

## 📊 Headline Results

- 🚀 **Hybrid model AUC ≈ 0.94** (numeric + temporal + LDA + neural embeddings) vs **baseline AUC ≈ 0.72**.
- 📉 **Closing restaurants drift downward in sentiment** over months and years — confirmed by two independent statistical tests (Mann–Whitney *p* ≈ 2.6 × 10⁻⁹, Mann–Kendall-derived test *p* ≈ 1.8 × 10⁻⁶).
- 👥 **Reviewer composition matters.** Restaurants whose review base is dominated by Yelp super-users + network-central reviewers close at **53.3%** vs the **39.8%** baseline (+13.5 pp on n = 884), and the riskiest top decile of restaurants close **1.37× more often** than average.

The full story, charts, and methodology live in [`main_notebook.ipynb`](./main_notebook.ipynb).

---

## 🌐 Data

- **Dataset:** [Yelp Open Dataset](https://www.yelp.com/dataset)
- **Files used:** `yelp_academic_dataset_business.json`, `yelp_academic_dataset_review.json`, `yelp_academic_dataset_user.json`
- **Geographic scope:** Philadelphia restaurants only (filtered by `categories` containing "Restaurants")
- **Project cap:** first **300,000** matching reviews for reproducible runtimes
- **Class label:** `is_open` (binary; ~40% of restaurants in our snapshot are closed)

The dataset is **not** committed to this repository because of size and licensing. After downloading, place the files into a local `yelp_dataset/` folder at the project root:

```text
yelp_dataset/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
└── yelp_academic_dataset_user.json
```

### 🧹 Preprocessing summary

- Filter businesses to Philadelphia and `categories` containing "Restaurants".
- Stream reviews and keep only those tied to the filtered `business_id`s; cap at 300k for runtime.
- Stream `user.json` (~1.99 M users) and keep the ~124k users present in our review subsample (100% coverage).
- Text prep for RQ1: minimum-length filter, TF–IDF (1–2 grams), per-review LDA, sentence-transformer embeddings (`all-MiniLM-L6-v2`) + PCA-64.
- VADER sentiment scoring per review → quarterly aggregation per restaurant for RQ2 (≥4 quarters, ≥20 reviews).
- 15-feature reviewer profile (lifetime review count, elite years, friends, fans, compliments, account age, …) for RQ3 anomaly detection.

---

## 🛠️ How to reproduce

This project was developed locally in **Jupyter / VS Code** but runs cleanly in **Google Colab** as well. Recommended order:

1. **Download Yelp data** from <https://www.yelp.com/dataset> and place the three JSON files inside `yelp_dataset/` (see structure above).
2. **Set up the environment.** This project was developed and tested on **Python 3.11** (Google Colab default; also runs on 3.13 locally).
   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Some libraries (e.g. `gensim`, `bertopic`, `sentence-transformers`) are also installed inline at the top of `main_notebook.ipynb` via `%pip install`, so a fresh Colab kernel works out of the box.
3. **Run the notebooks** in this order:
   1. `checkpoints/checkpoint_1.ipynb` — initial EDA & dataset framing.
   2. `checkpoints/checkpoint_2.ipynb` — research-question formation, motivation, feasibility, smoke tests.
   3. `main_notebook.ipynb` — the full analysis and final results (this is the deliverable).

> 💡 **Colab tip:** mount Drive, drop the three `yelp_academic_dataset_*.json` files into a `yelp_dataset/` folder there, then point the `DATA_DIR` constant at it.

---

## 🌱 Key dependencies

The big-rock packages (full pin list lives in [`requirements.txt`](./requirements.txt)):

| Package | Version | Used for |
|---|---|---|
| `python` | **3.11** (Colab) / 3.13 (local) | runtime |
| `pandas` | 2.x | data wrangling |
| `numpy` | 2.1 | arrays / math |
| `scikit-learn` | 1.4+ | Logistic Regression, Isolation Forest, `HistGradientBoostingClassifier`, PCA, metrics |
| `matplotlib` / `seaborn` | latest | charts |
| `nltk` | 3.x | VADER sentiment |
| `gensim` | 4.x | LDA + coherence (c\_v) |
| `bertopic` | 0.16+ | neural topic modelling |
| `sentence-transformers` | 2.x | review embeddings (`all-MiniLM-L6-v2`) |
| `pymannkendall` | 1.x | Mann–Kendall trend test |
| `networkx` | 3.x | bipartite graph + PageRank |
| `scipy` | 1.14 | statistical tests (Mann–Whitney) |

---

## 🌳 Repository structure

```text
data_mining_project/
├── README.md                       # you are here 📍
├── main_notebook.ipynb             # 👈 the main deliverable
├── requirements.txt                # full pinned dependency list
├── checkpoints/
│   ├── checkpoint_1.ipynb          # initial EDA & dataset framing
│   └── checkpoint_2.ipynb          # research-question formation + smoke tests
├── assets/                         # 🖼️ exported charts & figures (PNG)
│   ├── 01_eda_4panel_overview.png
│   ├── 02_baseline_roc.png
│   ├── 03_rq1a_tfidf_topterms.png
│   ├── 04_rq1b_topic_models_auc.png
│   ├── 05_rq2_sentiment_trajectory.png
│   ├── 06_rq3_quartile_and_kde.png
│   ├── 07_rq3_reviewer_network.png
│   ├── 08_rq3_pagerank_x_anomaly_heatmap.png
│   ├── 09_rq3_risk_decile_and_lift.png
│   ├── 10_rq2_yearly_median_bootstrap_ci.png
│   ├── 11_rq2_slope_x_volume_atlas.png
│   ├── 12_rq3_super_user_radar.png
│   └── ckpt{1,2}_*.png             # checkpoint-notebook figures
└── yelp_dataset/                   # local data folder (gitignored, not on GitHub)
    ├── yelp_academic_dataset_business.json
    ├── yelp_academic_dataset_review.json
    └── yelp_academic_dataset_user.json
```

---

## 🎯 Results summary

> Three independent evidence streams — review text, sentiment trajectory, and reviewer-network behaviour each carry real closure signal beyond stars, and together they push closure prediction from **AUC ≈ 0.72** (numeric baseline) to **AUC ≈ 0.94** (hybrid). Closing restaurants drift downward in sentiment over time (*p* < 10⁻⁶), and restaurants disproportionately reviewed by Yelp super-users close **+13.5 pp** more often than baseline. The combined signal is more *explainable* and more *forward-looking* than the star average alone, exactly the kind of layered evidence a platform, owner, or investor would want.

Full methodology, charts, statistical tests, limitations, and future work are documented in [`main_notebook.ipynb`](./main_notebook.ipynb).
