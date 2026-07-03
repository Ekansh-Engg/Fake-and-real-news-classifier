# Twitter Airline Sentiment Analysis (NLP)

A sentiment classification project on real-world, messy Twitter data — built as a first hands-on NLP project after completing text preprocessing, POS tagging, NER, and sentiment analysis fundamentals.

## Dataset

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (Figure Eight, via Kaggle) — 14,640 tweets directed at major US airlines, labeled positive / neutral / negative.

Chosen deliberately over cleaner datasets because the raw text includes mentions, hashtags, URLs, contractions, and inconsistent casing — realistic preprocessing challenges rather than pre-cleaned toy data.

## Project Status

- [x] Data exploration & class balance check
- [x] Text preprocessing pipeline
- [x] Vectorization (TF-IDF)
- [x] Model training & evaluation (Logistic Regression vs Naive Bayes)
- [x] Error analysis (confusion matrix)
- [ ] VADER baseline comparison (optional stretch goal)

## Key Preprocessing Decisions

- **Class imbalance**: Negative tweets make up 63% of the dataset vs. 16% positive — noted early since it affects metric choice (F1 over accuracy) and model training later, not just preprocessing.
- **Mentions (`@airline`)**: Stripped entirely — present in 100% of tweets, structural rather than sentiment-bearing.
- **Hashtags**: Kept the word, dropped the `#` symbol (`#neveragain` → `neveragain`) since hashtags can carry real sentiment.
- **Contractions**: Expanded before punctuation removal (`didn't` → `did not`) to prevent negation words from being mangled into meaningless fragments like `didn` + `t`.
- **Negation words**: Removed `not`, `no`, `never`, etc. from the stopword list — negation flips sentiment meaning and is critical signal for this task.
- **Known limitation**: Bag-of-words style preprocessing can't capture sarcasm or mid-tweet sentiment shifts (e.g. "I didn't like it... but NOW I do!"). Flagged as a limitation, not silently ignored.

## Results

Two models were trained on TF-IDF features (5000 features, unigrams + bigrams) and compared:

| Model | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---|---|---|---|
| Logistic Regression (class_weight='balanced') | 0.77 | 0.72 | 0.85 | 0.60 | 0.71 |
| Multinomial Naive Bayes | 0.75 | 0.61 | 0.84 | 0.42 | 0.58 |

**Logistic Regression was chosen as the better model**, despite a similar overall accuracy to Naive Bayes. The key difference shows up in per-class recall, not the headline accuracy number:

- Naive Bayes achieved 0.99 recall on the negative class but only 0.29 recall on neutral — a sign it defaulted toward the majority class (63% of the dataset) rather than genuinely learning to distinguish sentiment.
- Logistic Regression, using `class_weight='balanced'`, performed far more evenly across all three classes (0.81 / 0.66 / 0.74 recall for negative/neutral/positive respectively).

This is a deliberate illustration of why accuracy alone is a misleading metric on imbalanced datasets — both models look similar at a glance (0.77 vs 0.75), but the per-class breakdown reveals a meaningfully different quality of model underneath.

### Confusion Matrix Insight (Logistic Regression)

Misclassifications are concentrated between adjacent sentiment classes (negative↔neutral, neutral↔positive) rather than opposite extremes (negative↔positive) — this suggests the model has learned a genuine sentiment gradient rather than arbitrary noise. Neutral remains the hardest class to classify, which is expected: neutral tweets lack the strong, distinctive vocabulary that negative/positive tweets tend to have.

### Known Limitations

- Bag-of-words/TF-IDF approaches cannot capture sarcasm or sentiment shifts within a single tweet (e.g. "I didn't like it... but NOW I do!").
- Neutral class remains the weakest performer across both models — likely an inherent difficulty of the class itself, not a fixable preprocessing issue.

## Setup

```bash
git clone https://github.com/Ekansh-Engg/twitter_sentiment_nlp.git
cd twitter_sentiment_nlp
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

Download `Tweets.csv` from the [Kaggle dataset link above](#dataset) and place it in `data/`.

## Project Structure

```
├── data/            # raw + cleaned data (gitignored)
├── notebooks/       # exploration, preprocessing & modeling notebooks
├── src/             # reusable scripts (planned)
├── requirements.txt
└── README.md
```