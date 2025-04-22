from torch.utils.data import Dataset
import torch
from datasets import load_dataset


class SimpleSentimentDataset(Dataset):
    def __init__(self, text, sentiment, tokinizer):
        self.text = text
        self.sentiment = sentiment
        self.tokinizer = tokinizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
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
        return len(self.text)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        data[self.text_id] = torch.tensor(self.tokenizer(data[self.text_id]))
        return data, torch.tensor(self.sentiment[idx])


def get_poem_sentiment_dataset(tokenizer, path=None):
    if path:
        raw_dataset = load_dataset(path)
    else:
        raw_dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_data, validation_data, test_data = (
        raw_dataset["train"],
        raw_dataset["validation"],
        raw_dataset["test"],
    )
    return (
        SimpleSentimentDataset(
            train_data["verse_text"], train_data["label"], tokenizer
        ),
        SimpleSentimentDataset(
            validation_data["verse_text"], validation_data["label"], tokenizer
        ),
        SimpleSentimentDataset(test_data["verse_text"], test_data["label"], tokenizer),
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
        SimpleSentimentDataset(train_data["text"], train_data["sentiment"], tokenizer),
        SimpleSentimentDataset(test_data["text"], test_data["sentiment"], tokenizer),
    )


if __name__ == "__main__":
    train_data, test_data = get_basic_tweet_sentiment_dataset(None)
    print(len(train_data))
    print(len(test_data))
