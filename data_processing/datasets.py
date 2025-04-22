from torch.utils.data import Dataset
import torch


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
