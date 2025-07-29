import pandas as pd
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin
from wordcloud import STOPWORDS
import fasttext
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm

tqdm.pandas()
stopwords = set(STOPWORDS)


EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emojis
    "\U0001f300-\U0001f5ff"  # symbols and pictograms
    "\U0001f680-\U0001f6ff"  # transport
    "\U0001f1e0-\U0001f1ff"  # flags
    "]+",
    flags=re.UNICODE,
)


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps numerical labels to their corresponding text labels in the given DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing a column named 'label'
                           with numerical values representing different categories.

    Returns:
        pd.DataFrame: The original DataFrame with an additional column 'label_text'
                       that contains the mapped text labels.
                       The mapping is as follows:
                       - 0: "Human"
                       - 1: "Bot"
    """
    label_map = {0: "Human", 1: "Bot"}
    df["label_text"] = df["label"].map(label_map)
    return df


class TweetExploder(BaseEstimator, TransformerMixin):
    """
    A transformer that explodes a DataFrame column containing lists of tweets into separate rows.

    Methods:
        fit(X, y=None):
            Fits the transformer to the data (no operation in this case).

        transform(X, y=None):
            Transforms the input DataFrame by exploding the 'tweet' column into separate rows.

    Parameters:
        X (pd.DataFrame): Input DataFrame containing a column named 'tweet' with lists of tweets.
        y (optional): Ignored. Exists for compatibility with scikit-learn's pipeline.

    Returns:
        pd.DataFrame: A new DataFrame with the 'tweet' column exploded into separate rows.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None) -> "TweetExploder":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_copy = X.copy()
        return X_copy.explode("tweet").reset_index(drop=True)


class TweetFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A transformer to extract features from tweets.

    This transformer adds several new columns to the input DataFrame based on the content of the 'tweet' column.
    The new columns include:
        - word_count: The number of words in the tweet.
        - character_count: The total number of characters in the tweet.
        - hashtag_count: The number of hashtags present in the tweet.
        - mention_count: The number of mentions (user tags) in the tweet.
        - link_count: The number of links present in the tweet.
        - emoji_count: The number of emojis present in the tweet.

    Attributes:
        None

    Methods:
       fit(X, y=None):
            Fits the transformer to the data (no operation in this case).

        transform(X):
            Transforms the input DataFrame by adding new feature columns.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None) -> "TweetFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        # word count
        X_copy["word_count"] = X_copy["tweet"].apply(lambda tweet: len(tweet.split()))

        # character count
        X_copy["character_count"] = X_copy["tweet"].apply(lambda tweet: len(tweet))

        # hashtag count
        X_copy["hashtag_count"] = X_copy["tweet"].apply(lambda tweet: tweet.count("#"))

        #  @ count
        X_copy["mention_count"] = X_copy["tweet"].apply(lambda tweet: tweet.count("@"))

        # links count
        X_copy["link_count"] = X_copy["tweet"].apply(lambda tweet: tweet.count("https"))

        # emoji count
        X_copy["emoji_count"] = X_copy["tweet"].apply(
            lambda tweet: len(EMOJI_PATTERN.findall(tweet))
        )

        return X_copy


class TweetProcessor(BaseEstimator, TransformerMixin):
    """
    A transformer to preprocess tweets.

    Attributes:
        None

    Methods:
        fit(X, y=None):
            Fits the transformer to the data (does nothing in this case).

        transform(X):
            Preprocesses the tweets in the provided DataFrame and returns a new DataFrame
            with the cleaned tweets.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "TweetProcessor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        # lowercase
        X_copy["tweet"] = X_copy["tweet"].str.lower()

        # remove links -> [LINK]
        X_copy["tweet"] = X_copy["tweet"].apply(
            lambda tweet: re.sub(r"http\S+|www\S+|https\S+", "[LINK]", tweet)
        )
        # remove mentions -> [USER]
        X_copy["tweet"] = X_copy["tweet"].apply(
            lambda tweet: re.sub(r"@\w+", "[USER]", tweet)
        )

        # remove '#' from hashtags
        X_copy["tweet"] = X_copy["tweet"].apply(
            lambda tweet: re.sub(r"#(\w+)", r"\1", tweet)
        )

        # remove emojis -> [EMOJI]
        X_copy["tweet"] = X_copy["tweet"].apply(
            lambda tweet: EMOJI_PATTERN.sub("[EMOJI]", tweet)
        )

        # remove punctuation
        punctuation_pattern = f"[{re.escape(string.punctuation)}]"
        X_copy["tweet"] = X_copy["tweet"].apply(
            lambda tweet: re.sub(punctuation_pattern, "", tweet)
        )

        # remove stop words
        X_copy["tweet"] = X_copy["tweet"].apply(
            lambda tweet: " ".join(
                [word for word in tweet.split() if word not in stopwords]
            )
        )

        return X_copy


class LanguageDetector(BaseEstimator, TransformerMixin):
    """
    A transformer for detecting tweet language using fastText.

    Attributes:
        model (fasttext.FastText): The fastText model loaded for language detection.

    Methods:
        fit(X, y=None):
            Fits the transformer. This method does not perform any fitting as the model is pre-trained.

        transform(X):
            Transforms the input DataFrame by adding a new column 'language' that contains
            the detected language for each tweet.
    """

    def __init__(self) -> None:
        super().__init__()
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification", filename="model.bin"
        )
        self.model = fasttext.load_model(model_path)

    def fit(self, X: pd.DataFrame, y=None) -> "LanguageDetector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        X_copy["language"] = X_copy["tweet"].progress_apply(
            lambda tweet: (
                self.model.predict(tweet.replace("\n", " ").replace("\r", " "))[0][
                    0
                ].replace("__label__", "")
                if isinstance(tweet, str) and tweet.strip()
                else "unknown"
            )
        )

        return X_copy


class EnglishSelector(BaseEstimator, TransformerMixin):
    """
    A transformer that selects rows from a DataFrame where the 'language' column
    is equal to 'eng_Latn'.

    Attributes:
        None

    Methods:
        fit(X, y=None):
            Fits the transformer. This method does not perform any operation
            and simply returns the instance.

        transform(X):
            Transforms the input DataFrame by selecting only the rows where
            the 'language' column is 'eng_Latn'.

    Parameters:
        X (pd.DataFrame): The input DataFrame containing a 'language' column.
        y (optional): Ignored, exists for compatibility with scikit-learn.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows with 'language'
        equal to 'eng_Latn'.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None) -> "EnglishSelector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        return X_copy[X_copy["language"] == "eng_Latn"]


class SentimentScorer(BaseEstimator, TransformerMixin):
    """
    A transformer for scoring the sentiment of tweets using a pre-trained model.

    This class utilizes the `cardiffnlp/twitter-xlm-roberta-base-sentiment` model
    to classify tweets into three sentiment categories: negative, neutral, and positive.

    Attributes:
        model: A pre-trained sequence classification model.
        tokenizer: A tokenizer for processing input text.

    Methods:
        fit(X, y=None):
            Fits the transformer to the data (no operation in this case).

        transform(X):
            Transforms the input DataFrame by adding a 'sentiment' column
            with the predicted sentiment for each tweet.
    """

    def __init__(self) -> None:
        super().__init__()
        model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def fit(self, X: pd.DataFrame, y=None) -> "SentimentScorer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        def get_sentiment(tweet):
            if not isinstance(tweet, str) or tweet.strip() == "":
                return "neutral"

            inputs = self.tokenizer(
                tweet,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                label = torch.argmax(probs).item()

                # negative = 0, neutral = 1, positive = 2
                labels = ["negative", "neutral", "positive"]
                return labels[label]

        X_copy["sentiment"] = X_copy["tweet"].progress_apply(get_sentiment)
        return X_copy
