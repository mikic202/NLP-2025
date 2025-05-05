import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_advanced_tweet_sentiment_data(
    tweet_data: pd.DataFrame, means=None, stds=None
):
    tweet_data.dropna(subset=["text", "sentiment"], inplace=True)
    label_encoder = LabelEncoder()
    tweet_data["sentiment"] = label_encoder.fit_transform(tweet_data["sentiment"])
    tweet_data["Time of Tweet"] = label_encoder.fit_transform(
        tweet_data["Time of Tweet"]
    )
    tweet_data["Country"] = label_encoder.fit_transform(tweet_data["Country"])
    tweet_data["Age of User"] = label_encoder.fit_transform(tweet_data["Age of User"])

    tweet_data = pd.get_dummies(
        tweet_data, columns=["Country", "Age of User", "Time of Tweet"]
    )

    numeric_columns = ["Population -2020", "Land Area (Km²)", "Density (P/Km²)"]
    if means is None or stds is None:
        means = tweet_data[numeric_columns].mean()
        stds = tweet_data[numeric_columns].std()

    tweet_data[numeric_columns] = (tweet_data[numeric_columns] - means) / stds
    tweet_data = tweet_data.reset_index(drop=True)

    return tweet_data, means, stds
