cat > README.md << 'EOF'
# Yelp Restaurant Success Analysis

## 📖 Overview
A data mining project analyzing the Yelp Open Dataset to uncover patterns in restaurant reviews, ratings, and business survival. The project applies multiple data mining techniques — including text mining, clustering, graph mining, anomaly detection, topic modeling, and sentiment analysis — to investigate what drives restaurant success and closure.

This project is part of a graduate-level Data Mining course.

## 📊 Dataset
- **Source**: [Yelp Open Dataset](https://www.yelp.com/dataset)
- **Size**: ~150K businesses, ~7M reviews across multiple JSON files
- **Scope**: Initial EDA focuses on Philadelphia restaurants (~7,575 businesses, ~300K reviews), with potential expansion to additional cities
- **Key Files Used**: `business.json`, `review.json`, `user.json`

> **Note**: The dataset is not included in this repository due to size and licensing constraints. Download it from [Yelp](https://www.yelp.com/dataset) and place the files in a `yelp_dataset/` directory.

## 🔍 Research Questions
1. Can latent topics in restaurant reviews predict whether a restaurant will close?
2. Do restaurants with distinct review sentiment trajectories have different survival outcomes?
3. Can clustering restaurants by review text features reveal distinct "success profiles"?
4. Do reviewer characteristics systematically influence ratings?
5. Can user-business interaction graphs reveal community structures or anomalous reviewing behavior?

## ⚙️ Techniques

| Technique | Type |
|-----------|------|
| TF-IDF + Document Clustering | Course (Text Mining, Clustering) |
| Topic Modeling (LDA/NMF) | Beyond-Course |
| Sentiment Analysis (VADER) | Beyond-Course |
| Classification (is_open prediction) | Course (Large-Scale ML) |
| Graph Mining (User-Business Network) | Course (Graph Mining) |
| Anomaly Detection on Reviewers | Course (Anomaly Detection) |

## 📁 Repository Structure

