from torch.utils.data import Dataset
import torch
from datasets import load_dataset
import kagglehub
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.preprocessing import preprocess_advanced_tweet_sentiment_data


class SimpleSentimentDataset(Dataset):
    def __init__(self, text, sentiment, tokinizer):
        self.text = text
        self.sentiment = sentiment
        self.tokinizer = tokinizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if self.tokinizer is None:
            return self.text[idx], self.sentiment[idx]
        return torch.tensor(self.tokinizer(self.text[idx])), torch.tensor(
            self.sentiment[idx]
        )


class AdditionalDataSentimentDataset(Dataset):
    def __init__(self, data, sentiment, tokenizer, text_id: str = "text"):
        self.data = data
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.text_id = text_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.tokenizer is None:
            return self.data.iloc[idx], self.sentiment[idx]
        data = self.data.iloc[idx]
        data[self.text_id] = torch.tensor(self.tokenizer(data[self.text_id]))
        return data, torch.tensor(self.sentiment[idx])


def get_poem_sentiment_dataset(tokenizer, path=None):
    if path:
        raw_dataset = load_dataset(path)
    else:
        raw_dataset = load_dataset("google-research-datasets/poem_sentiment")
    label_encoder = LabelEncoder()
    train_data, validation_data, test_data = (
        raw_dataset["train"],
        raw_dataset["validation"],
        raw_dataset["test"],
    )
    return (
        SimpleSentimentDataset(
            train_data["verse_text"],
            label_encoder.fit_transform(train_data["label"]),
            tokenizer,
        ),
        SimpleSentimentDataset(
            validation_data["verse_text"],
            label_encoder.fit_transform(validation_data["label"]),
            tokenizer,
        ),
        SimpleSentimentDataset(
            test_data["verse_text"],
            label_encoder.fit_transform(test_data["label"]),
            tokenizer,
        ),
    )


def get_basic_tweet_sentiment_dataset(tokenizer, path=None):
    if path:
        raw_dataset = load_dataset(path)
    else:
        raw_dataset = load_dataset("stanfordnlp/sentiment140")
    train_data, test_data = (
        raw_dataset["train"],
        raw_dataset["test"],
    )
    return (
        SimpleSentimentDataset(
            train_data["text"],
            train_data["sentiment"],
            tokenizer,
        ),
        SimpleSentimentDataset(
            test_data["text"],
            test_data["sentiment"],
            tokenizer,
        ),
    )


def get_advanced_tweet_sentiment_dataset(tokenizer, path=None):
    if not path:
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
    train_data, means, stds = preprocess_advanced_tweet_sentiment_data(
        pd.read_csv(path + "/train.csv", encoding="ISO-8859-1", engine="python")
    )
    train_data["text"] = train_data["text"].fillna("")
    test_data, _, _ = preprocess_advanced_tweet_sentiment_data(
        pd.read_csv(path + "/test.csv", encoding="ISO-8859-1", engine="python"),
        means,
        stds,
    )
    test_data["text"] = test_data["text"].fillna("")
    return (
        AdditionalDataSentimentDataset(
            train_data.drop(["sentiment", "selected_text", "textID"], axis=1),
            train_data["sentiment"],
            tokenizer,
            text_id="text",
        ),
        AdditionalDataSentimentDataset(
            test_data.drop(["sentiment", "textID"], axis=1),
            test_data["sentiment"],
            tokenizer,
            text_id="text",
        ),
        means,
        stds,
    )
