import pandas as pd
import re
import string
from nltk.corpus import opinion_lexicon
from sklearn.base import BaseEstimator, TransformerMixin
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

POSITIVE_WORDS = set(opinion_lexicon.positive())
NEGATIVE_WORDS = set(opinion_lexicon.negative())

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emojis
    "\U0001f300-\U0001f5ff"  # symbols and pictograms
    "\U0001f680-\U0001f6ff"  # transport
    "\U0001f1e0-\U0001f1ff"  # flags
    "]+",
    flags=re.UNICODE,
)


class TweetFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A class for extracting features from tweets.

    """

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None) -> "TweetFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds new columns ('avg_word_count', 'avg_character_count', 'avg_hashtag_count', 'avg_mention_count',
        'avg_link_count', 'avg_emoji_count', 'avg_positive_word_count', 'avg_negative_word_count') to X based on column 'tweet'.

        Args:
            X (pd.DataFrame): A pandas DataFrame containing column 'tweet'.

        Returns:
            pd.DataFrame: A modified X with new added columns.
        """
        X_copy = X.copy()
        X_copy["tweets_joined"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                " [SEP] ".join(str(t) for t in tweet_list)
                if isinstance(tweet_list, list)
                else ""
            )
        )

        # average word count
        X_copy["avg_word_count"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                sum(len(tweet.split()) for tweet in tweet_list) / len(tweet_list)
                if isinstance(tweet_list, list) and len(tweet_list) > 0
                else 0
            )
        )

        # average character count
        X_copy["avg_character_count"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                sum(len(tweet) for tweet in tweet_list) / len(tweet_list)
                if isinstance(tweet_list, list) and len(tweet_list) > 0
                else 0
            )
        )

        # average hashtag count
        X_copy["avg_hashtag_count"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                sum(tweet.count("#") for tweet in tweet_list) / len(tweet_list)
                if isinstance(tweet_list, list) and len(tweet_list) > 0
                else 0
            )
        )

        # average @ count
        X_copy["avg_mention_count"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                sum(tweet.count("@") for tweet in tweet_list) / len(tweet_list)
                if isinstance(tweet_list, list) and len(tweet_list) > 0
                else 0
            )
        )

        # average links count
        X_copy["avg_link_count"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                sum(tweet.count("https") for tweet in tweet_list) / len(tweet_list)
                if isinstance(tweet_list, list) and len(tweet_list) > 0
                else 0
            )
        )

        # average emoji count
        X_copy["avg_emoji_count"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                sum(len(EMOJI_PATTERN.findall(tweet)) for tweet in tweet_list)
                / len(tweet_list)
                if isinstance(tweet_list, list) and len(tweet_list) > 0
                else 0
            )
        )

        # average positive words count
        X_copy["avg_positive_word_count"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                sum(
                    sum(1 for w in tweet.split() if w.lower() in POSITIVE_WORDS)
                    for tweet in tweet_list
                )
                / len(tweet_list)
                if isinstance(tweet_list, list) and len(tweet_list) > 0
                else 0
            )
        )

        # average negative words count
        X_copy["avg_negative_word_count"] = X_copy["tweet"].apply(
            lambda tweet_list: (
                sum(
                    sum(1 for w in tweet.split() if w.lower() in NEGATIVE_WORDS)
                    for tweet in tweet_list
                )
                / len(tweet_list)
                if isinstance(tweet_list, list) and len(tweet_list) > 0
                else 0
            )
        )
        return X_copy


class TweetProcessor(BaseEstimator, TransformerMixin):
    """
    A class for preprocessing tweets.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "TweetProcessor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses tweets (changes to lowercase, removes links, emojis, hashtag symbol, removes punctuation marks etc.).

        Args:
            X (pd.DataFrame): A pandas DataFrame containing column 'tweets_joined'.

        Returns:
            pd.DataFrame: A pandas DataFrame with preprocessed tweets.
        """
        X_copy = X.copy()
        # lowercase
        X_copy["tweets_joined"] = X_copy["tweets_joined"].str.lower()

        # remove links -> [LINK]
        X_copy["tweets_joined"] = X_copy["tweets_joined"].apply(
            lambda tweet: re.sub(r"http\S+|www\S+|https\S+", "[LINK]", tweet)
        )
        # remove mentions -> [USER]
        X_copy["tweets_joined"] = X_copy["tweets_joined"].apply(
            lambda tweet: re.sub(r"@\w+", "[USER]", tweet)
        )

        # remove '#' from hashtags
        X_copy["tweets_joined"] = X_copy["tweets_joined"].apply(
            lambda tweet: re.sub(r"#(\w+)", r"\1", tweet)
        )

        # remove emojis -> [EMOJI]
        X_copy["tweets_joined"] = X_copy["tweets_joined"].apply(
            lambda tweet: EMOJI_PATTERN.sub("[EMOJI]", tweet)
        )

        # remove punctuation
        punctuation_pattern = f"[{re.escape(string.punctuation)}]"
        X_copy["tweets_joined"] = X_copy["tweets_joined"].apply(
            lambda tweet: re.sub(punctuation_pattern, "", tweet)
        )

        # remove stop words
        X_copy["tweets_joined"] = X_copy["tweets_joined"].apply(
            lambda tweet: " ".join(
                [word for word in tweet.split() if word not in stopwords]
            )
        )

        return X_copy["tweets_joined"].tolist()
