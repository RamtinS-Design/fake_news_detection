# Fake News Detection

Text classification project to flag news articles as fake or real, based on the Kaggle "Fake and Real News" dataset. This is a work in progress: the data pipeline and text cleaning are done, the classifier itself isn't built yet.

## Status

- [x] Load and inspect the data
- [x] Merge and label the two source files
- [x] Clean and normalize article text
- [ ] Tokenize / vectorize for modeling
- [ ] Train/test split
- [ ] Train a classifier
- [ ] Evaluate

## Dataset

Two CSVs from the Kaggle Fake and Real News Dataset:

- `Fake.csv` — 23,481 articles
- `True.csv` — 21,417 articles

Both share the same four columns: `title`, `text`, `subject`, `date`. Neither file has missing values. `Fake.csv` has 3 duplicate rows and `True.csv` has 206 — identified but not yet dropped, which is worth doing before any train/test split so duplicates don't leak across the two sets.

Fake articles are labeled `0` and real articles `1`, then the two frames are concatenated and shuffled into a single 44,898-row dataset.

## Preprocessing

`title` and `text` are joined into one `content` field, then cleaned with a custom function:

- lowercased
- URLs and HTML tags stripped
- digits and punctuation removed
- extra whitespace collapsed
- tokenized with NLTK, stopwords removed, and remaining words lemmatized

The cleaned result is written out to `cleaned_fake_news.csv`, and `X` (content) / `y` (label) are split out for modeling.

## How to Run it

Needs `Fake.csv` and `True.csv` (from the Kaggle Fake and Real News Dataset) in the working directory.
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv

```
pip install pandas numpy nltk tensorflow
```

## What's next

The notebook currently ends right after importing `Tokenizer` and `pad_sequences` from `tensorflow.keras`, which points toward a sequence-based model (e.g. an embedding layer feeding an LSTM) rather than a classic bag-of-words classifier. Remaining work:

- Drop the duplicate rows found during inspection
- Fit the tokenizer on the cleaned text and pad sequences to a fixed length
- Split into train/test sets
- Build and train the model
- Evaluate with accuracy, precision, recall, F1, and a confusion matrix — same as would be expected for any binary classifier

