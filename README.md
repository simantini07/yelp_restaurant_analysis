# Yelp Restaurant Survival Signals (Philadelphia)

This project studies a practical question: **why do some restaurants close even when star ratings still look fine?**  
Using Yelp Open Dataset data for Philadelphia restaurants, the analysis compares baseline numeric features against richer signals from review text, sentiment trends over time, and reviewer behavior/network structure.

👉 **Start here: `main_notebook.ipynb`**

🎥 **Project video:** (https://www.youtube.com/watch?v=V9y9EpvLQzc)

## Research Questions

1. **RQ1 (Text signal):** Does review text (TF-IDF and topic features) improve closure prediction beyond numeric baseline features?
2. **RQ2 (Trajectory signal):** Do quarterly sentiment trajectories differ between open and closed restaurants?
3. **RQ3 (Reviewer/network signal):** Do anomalous reviewer exposure and graph-based reviewer centrality align with higher closure risk?

## Data

- **Dataset:** [Yelp Open Dataset](https://www.yelp.com/dataset)
- **Files used:** business, review, and user-level information (as needed by the notebook pipeline)
- **Geographic scope:** Philadelphia restaurants only
- **Project cap:** first 300,000 matching reviews for computational reproducibility

The dataset is not committed to this repo because of size and licensing.  
After downloading, place files under:

`yelp_dataset/`

Expected main files include:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`

## Preprocessing Summary

- Filter businesses to Philadelphia + categories containing Restaurant
- Keep only reviews tied to filtered restaurant `business_id`s
- Cap matching reviews for stable runtime
- Text preparation for RQ1 modeling (minimum length filters, TF-IDF/topic features)
- Sentiment scoring per review (VADER with star-based fallback), then quarterly aggregation per business
- Reviewer feature engineering for anomaly detection and graph aggregation

## Reproducibility

This project was developed in a notebook workflow and is reproducible from this repository.

1. Download Yelp data and place it in `yelp_dataset/` as described above.
2. Create and activate a Python environment.
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Open and run:
   - `main_notebook.ipynb` (primary deliverable)
5. Optional progression notebooks:
   - `checkpoints/checkpoint_1.ipynb`
   - `checkpoints/checkpoint_2.ipynb`

## Key Dependencies

- Python `3.13.5`
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk (VADER)
- statsmodels
- pymannkendall
- gensim
- bertopic
- networkx

For exact versions, see `requirements.txt`.

## Repository Structure

```text
data_mining_project/
├── main_notebook.ipynb
├── requirements.txt
├── README.md
├── checkpoints/
│   ├── checkpoint_1.ipynb
│   └── checkpoint_2.ipynb
├── assets/
│   ├── 01_city_distribution.png
│   └── 02_star_distributions.png
├── scripts/
│   └── patch_rq_extensions.py
├── yelp_dataset/                  # local data folder (not for GitHub commit)
├── 837005056_project.ipynb        # original notebook snapshot
├── 837005056_project_ck2.ipynb    # original notebook snapshot
└── 837005056_project_final_submission.ipynb
```

## Results Summary

- Numeric baseline provides strong closure prediction performance.
- Text/topic features improve interpretability and provide complementary signal, but do not always outperform baseline AUC in every run.
- Sentiment trajectory tests show meaningful open-vs-closed trend differences.
- Reviewer anomaly exposure and graph-based reviewer centrality provide additional risk-pattern context for closure analysis.
