# Fake News Classifier (NLP)

A text classification project to detect fake vs. real news articles — built as a second NLP portfolio project, focused on catching real-world data leakage issues rather than just running a standard pipeline.

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle, by Clément Bisaillon) — ~44,900 news articles combined from two files (`Fake.csv`, `True.csv`), labeled Fake News / Real News.

## Project Status

- [x] Data exploration & merging
- [x] Data leakage detection & fixes
- [x] Text preprocessing pipeline
- [x] Vectorization (TF-IDF)
- [x] Model training & evaluation
- [x] Error analysis

## Key Data Quality & Leakage Findings

This dataset has several well-known but easy-to-miss traps that would let a model "cheat" instead of genuinely learning to detect fake news. Each was identified and fixed before modeling:

- **`subject` column leakage**: Real articles only ever use `politicsNews` or `worldnews` as their subject, while fake articles use six entirely different categories (`News`, `politics`, `Government News`, `left-news`, `US_News`, `Middle-east`) with zero overlap. Included as a feature, this column alone would let a model predict the label with near-perfect accuracy without learning anything about article content. **Dropped `subject` and `date` entirely** — only `title` and `text` are used.

- **Reuters dateline leakage**: 99.8% of real articles begin with a wire-service dateline (e.g. `"WASHINGTON (Reuters) - "`) versus just 1.4% of fake articles. Left in place, "Reuters" would become a near-perfect single-word predictor of the label. Fixed with a regex that strips the dateline prefix from the start of each article; remaining in-body mentions (~5,000 articles, legitimate references like "according to Reuters") are handled via a custom stopword addition rather than aggressive regex chasing.

- **Empty articles**: 631 articles had zero-length text after scraping — 630 of them labeled Fake News, all appearing to be embedded video/poll content (titles like "WATCH TUCKER CARLSON...", "TAKE OUR POLL...") rather than real text articles. Dropped rather than imputed, since empty text provides no learnable signal.

- **Broken contractions**: ~23% of articles (10,201) had contractions stored with a space instead of an apostrophe (`"couldn t"` instead of `"couldn't"`), a scraping artifact. Left unfixed, this silently destroys negation signal — `"couldn t wish"` would clean down to just `"wish"`, flipping the apparent sentiment/meaning. Fixed with a regex repair step that reconstructs the apostrophe before contraction expansion.

## Key Preprocessing Decisions

- **Lowercasing** applied for consistency across all text.
- **Contractions expanded** (`didn't` → `did not`) before punctuation removal, to prevent negation words from being mangled into meaningless fragments.
- **Negation words preserved**: `not`, `no`, `never`, etc. removed from the stopword list, since negation is critical signal for meaning (e.g. "did not confirm" vs. "confirmed").
- **Lemmatization** applied to reduce words to root form.
- No `@mention`/`#hashtag` handling needed (unlike the Twitter project) since this is news article text, not social media.

## Exploratory Data Analysis

After cleaning, the dataset was explored to understand class patterns before modeling. Initial investigative EDA (checking for leakage, sampling raw text) was done during preprocessing — see notebook 1. This section covers descriptive EDA on the cleaned data — see `notebooks/02_eda.ipynb`.

**Class balance**: ~51.6% Fake News / ~48.4% Real News — close to balanced, no significant class weighting needed for modeling.

**Article length**: Real News articles cluster tightly around shorter lengths (peaking ~0-100 words) and drop off quickly, while Fake News articles peak later (~200-250 words) with a wider, flatter spread. This matches the higher variance seen in summary statistics (std: 408 words for fake vs. 274 for real) — fake articles are both longer on average and less consistent in length, possibly reflecting more embellished or padded writing style versus concise wire-service reporting.

**Top words by class**: Heavy overlap on generic political terms (trump, president, state) is expected, since the dataset is politics-heavy regardless of label. However, a qualitative tone difference emerges: Fake News skews toward casual, personality-driven language (like, say, clinton, obama), while Real News skews toward formal, institutional vocabulary (government, party, election, official, told) — suggesting a difference in writing style/register beyond just word choice.

**Named entity comparison** (spaCy `en_core_web_sm`, sampled 3,000 articles per class): Fake News entities are dominated by individual political figures (Trump, Obama, Clinton) with minimal geopolitical diversity. Real News shows a broader geopolitical spread (U.S., Russia, China) alongside political figures, consistent with the `subject` categories (`politicsNews`, `worldnews`) associated with real articles. Note: spaCy occasionally misclassifies "Trump" as `ORG` rather than `PERSON` — a known ambiguity in political text NER, kept as-is rather than manually corrected to reflect genuine model behavior.

## Dataset Stats (post-cleaning)

- ~44,150 articles after removing empty/junk rows
- Roughly balanced classes: ~51.6% Fake News / ~48.4% Real News
- Fake articles average slightly longer (423 words) than real articles (382 words), with notably higher variance (std 408 vs. 274) — suggesting less consistency in fake article length/structure compared to standard journalistic format.

## Setup

```bash
git clone https://github.com/<your-username>/fake-news-classifier.git
cd fake-news-classifier
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

Download `Fake.csv` and `True.csv` from the [Kaggle dataset link above](#dataset) and place them in `data/`.

## Project Structure

```
├── data/            # raw + cleaned data (gitignored)
├── notebooks/       # exploration, preprocessing & modeling notebooks
├── src/             # reusable scripts (planned)
├── requirements.txt
└── README.md
```
## Results

Two models were trained on TF-IDF features (5000 features, unigrams + bigrams) after fixing three separate data leakage issues (see below):

| Model | Accuracy | Macro F1 | Fake News F1 | Real News F1 |
|---|---|---|---|---|
| Logistic Regression (class_weight='balanced') | 0.98 | 0.98 | 0.98 | 0.98 |
| Multinomial Naive Bayes | 0.93 | 0.93 | 0.93 | 0.93 |

**Logistic Regression was chosen as the better model.** Unlike the imbalanced Twitter sentiment project, this dataset is close to balanced (51.6%/48.4%), so both models perform evenly across classes here — Naive Bayes doesn't show the majority-class bias seen previously, confirming that the earlier NB weakness was a symptom of class imbalance specifically, not a general limitation of the algorithm.

### Data Leakage: Three Issues Found and Fixed

This dataset has several well-documented but easy-to-miss traps that would let a model shortcut the task instead of genuinely learning to detect fake news:

1. **`subject` column**: Real articles used only 2 subject categories (`politicsNews`, `worldnews`), fake articles used 6 entirely different ones, with zero overlap — a near-perfect label predictor if included. Dropped `subject` and `date` entirely.
2. **Reuters wire dateline**: 99.8% of real articles began with `"CITY (Reuters) - "`. Fixed via regex prefix stripping, with residual in-body mentions handled through a custom stopword.
3. **Media credit-line boilerplate**: ~36% of fake articles contained scraped photo/video credit text (e.g. `"Featured image via [name]/Getty Images"`) — an artifact of the source website's template, not article content. This was caught *after* initial modeling: "via" appeared as the single strongest predictor of Fake News (coefficient -12.0, larger than any genuine content word), which prompted a targeted investigation. Fixed via regex removal plus stopword additions (`via`, `getty`, `screenshot`) as a safety net for phrasing variants regex didn't catch.

**Validation**: after fixing all three leakage sources, model accuracy remained virtually unchanged (98.0% → 98.0% for Logistic Regression), which is a meaningful result in itself — it confirms the model was already relying primarily on genuine linguistic patterns, not the leaked shortcuts, even before the fixes. Top predictive features after the fix are journalistically meaningful (weekday mentions, attribution language like "said"/"told" for real news; informal references like "Mr.", political figures by surname for fake news) rather than formatting artifacts.

### Confusion Matrix

Errors are low and roughly symmetric: 135 Fake News articles misclassified as Real, 71 Real News misclassified as Fake (out of 8,837 test articles) — no strong bias toward either error direction.