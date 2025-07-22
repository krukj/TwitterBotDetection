import pandas as pd
import re
from nltk.corpus import opinion_lexicon

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


def add_new_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Adds new columns to DataFrame: avg_word_count, avg_character_count, avg_hashtag_count,
    avg_mention_count, avg_positive_word_count, avg_negative_word_count

    Args:
        df (pd.DataFrame): A pandas DataFrame containing column 'tweet'.

    Returns:
        pd.DataFrame: A pandas DataFrame with new columns.
    """
    df_copy = df.copy()

    # average word count
    df_copy["avg_word_count"] = df_copy["tweet"].apply(
        lambda tweet_list: (
            sum(len(tweet.split()) for tweet in tweet_list) / len(tweet_list)
            if isinstance(tweet_list, list) and len(tweet_list) > 0
            else 0
        )
    )

    # average character count
    df_copy["avg_character_count"] = df_copy["tweet"].apply(
        lambda tweet_list: (
            sum(len(tweet) for tweet in tweet_list) / len(tweet_list)
            if isinstance(tweet_list, list) and len(tweet_list) > 0
            else 0
        )
    )

    # average hashtag count
    df_copy["avg_hashtag_count"] = df_copy["tweet"].apply(
        lambda tweet_list: (
            sum(tweet.split().count("#") for tweet in tweet_list) / len(tweet_list)
            if isinstance(tweet_list, list) and len(tweet_list) > 0
            else 0
        )
    )

    # average @ count
    df_copy["avg_mention_count"] = df_copy["tweet"].apply(
        lambda tweet_list: (
            sum(tweet.split().count("@") for tweet in tweet_list) / len(tweet_list)
            if isinstance(tweet_list, list) and len(tweet_list) > 0
            else 0
        )
    )

    # average links count
    df_copy["avg_link_count"] = df_copy["tweet"].apply(
        lambda tweet_list: (
            sum(tweet.split().count("https") for tweet in tweet_list) / len(tweet_list)
            if isinstance(tweet_list, list) and len(tweet_list) > 0
            else 0
        )
    )

    # average emoji count
    df_copy["avg_emoji_count"] = df_copy["tweet"].apply(
        lambda tweet_list: (
            sum(len(EMOJI_PATTERN.findall(tweet)) for tweet in tweet_list)
            / len(tweet_list)
            if isinstance(tweet_list, list) and len(tweet_list) > 0
            else 0
        )
    )

    # average positive words count
    df_copy["avg_positive_word_count"] = df_copy["tweet"].apply(
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
    df_copy["avg_negative_word_count"] = df_copy["tweet"].apply(
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
    return df_copy
