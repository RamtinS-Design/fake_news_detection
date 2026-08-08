# Fake News Detection

Text classification project to flag news articles as fake or real, based on the Kaggle "Fake and Real News" dataset. This is a work in progress: the data pipeline and text cleaning are done, the classifier itself isn't built yet.

## Status

- [x] Load and inspect the data
- [x] Merge and label the two source files
- [x] Clean and normalize article text
- [x] Tokenize / vectorize for modeling
- [x] Train/test split
- [x] Train a classifier
- [x] Evaluate

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

## Modeling

The data is split 80/20 into train and test sets before any tokenizer fitting, so nothing from the test set leaks into the vocabulary.
The classifier is a Keras `Sequential` model:

- an embedding layer
- a bidirectional LSTM
- dropout and a small dense layer before the final sigmoid output

## Evaluation

The notebook reports accuracy, precision, recall, and F1 on the held-out test set, along with a full classification report and a confusion matrix heatmap. 

## How to Run it

Needs `Fake.csv` and `True.csv` (from the Kaggle Fake and Real News Dataset) in the working directory.
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv

```
pip install pandas numpy nltk tensorflow scikit-learn matplotlib seaborn
```

## What's next

- Compare against a simpler baseline (e.g. TF-IDF + logistic regression) to see whether the LSTM is actually earning its complexity
- Tune hyperparameters like embedding size, LSTM units, and dropout rate
- Try pretrained embeddings (GloVe or similar) instead of training the embedding layer from scratch
- Package the trained model behind a small script or API for inference on new articles

## Author

Made by (RamtinS-Design)
