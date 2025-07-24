import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import STOPWORDS
from collections import Counter

COLOR = "#4C72B0"
ALPHA = 0.8
PALETTE = ["#4C72B0", "#55A868"]
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 16
TICKS_FONTSIZE = 12
BAR_FONTSIZE = 12
ROTATION = 45
FIGSIZE = (10, 6)
SUBPLOTS_FIGSIZE = (14, 12)


def plot_label_distribution(df: pd.DataFrame) -> None:
    """
    Plots label distribution of df.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing column 'label' with values 0 and 1.
    """
    label_map = {0: "Human", 1: "Bot"}
    counts = df["label"].map(label_map).value_counts()
    total_count = sum(counts.values)
    _, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(
        counts.index, 100 * counts.values / total_count, alpha=ALPHA, color=COLOR
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=BAR_FONTSIZE,
        )

    ax.set_xlabel("Label", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Percentage (%)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Distribution of labels", fontsize=TITLE_FONTSIZE)
    plt.xticks(rotation=ROTATION, fontsize=TICKS_FONTSIZE)
    plt.tight_layout()
    plt.show()


def plot_most_common_words(
    df: pd.DataFrame, include_stopwords: bool = True, n: int = 10
) -> None:
    """
    Plots most common n words from the column 'tweet'.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing column 'tweet'.
        include_stopwords (bool, optional): Whether to include stop words or not. Defaults to True.
        n (int, optional): Number of most common words to be desplayed. Defaults to 10.
    """
    all_words = []
    stopwords = set(STOPWORDS)

    for tweet_list in df["tweet"]:
        if isinstance(tweet_list, list):
            for tweet in tweet_list:
                if isinstance(tweet, str):
                    words = tweet.lower().split()
                    if include_stopwords:
                        all_words.extend(words)
                    else:
                        filtered_words = [w for w in words if w not in stopwords]
                        all_words.extend(filtered_words)
    counter = Counter(all_words)
    most_common_words = counter.most_common(n)
    words, counts = zip(*most_common_words)

    plt.figure(figsize=FIGSIZE)
    bars = plt.bar(words, counts, color=COLOR, alpha=ALPHA)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            f"{yval}",
            ha="center",
            va="bottom",
            fontsize=BAR_FONTSIZE,
        )

    plt.title(
        f"Top {n} Most Common Words"
        + (" (with stop words)" if include_stopwords else " (without stop words)"),
        fontsize=TITLE_FONTSIZE,
    )
    plt.xlabel("Words", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Frequency", fontsize=LABEL_FONTSIZE)
    plt.xticks(rotation=ROTATION, fontsize=TICKS_FONTSIZE)
    plt.show()


def plot_histograms(df: pd.DataFrame, by_label: bool) -> None:
    """
    Plots histograms of 8 columns: avg_word_count, avg_character_count, avg_hashtag_count, avg_mention_count, avg_link_count,
    avg_emoji_count, avg_positive_word_count, avg_negative_word_count.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing columns 'avg_word_count', 'avg_character_count', 'avg_hashtag_count', 'avg_mention_count', 'avg_link_count',
        'avg_emoji_count', 'avg_positive_word_count', 'avg_negative_word_count' and 'label'.
        by_label (bool): Whether to group columns by label.
    """
    cols_to_plot = [
        "avg_word_count",
        "avg_character_count",
        "avg_hashtag_count",
        "avg_mention_count",
        "avg_link_count",
        "avg_emoji_count",
        "avg_positive_word_count",
        "avg_negative_word_count",
    ]
    _, axes = plt.subplots(4, 2, figsize=SUBPLOTS_FIGSIZE)
    if by_label:
        for col, ax in zip(cols_to_plot, axes.flatten()):
            sns.histplot(
                data=df,
                x=col,
                hue="label",
                bins=30,
                palette=PALETTE,
                edgecolor="black",
                alpha=ALPHA,
                common_norm=False,
                ax=ax,
            )
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Histogram of {col}")
    else:
        for col, ax in zip(cols_to_plot, axes.flatten()):
            sns.histplot(
                data=df,
                x=col,
                bins=30,
                edgecolor="black",
                alpha=ALPHA,
                common_norm=False,
                ax=ax,
            )
    plt.tight_layout()
    plt.show()


def plot_most_common_words_per_label(df: pd.DataFrame, label: str, n: int = 10) -> None:
    """
    Plots n most common words per label (bot or human).

    Args:
        df (pd.DataFrame): A pandas DataFrame containing columns 'tweet' and 'label'.
        label (str): Label to be displayed ('bot' or 'human').
        n (int, optional): Number of most common words to be displayed. Defaults to 10.
    """
    if label == "human":
        df = df[df["label"] == 0]
    elif label == "bot":
        df = df[df["label"] == 1]

    all_words = []
    stopwords = set(STOPWORDS)

    for tweet_list in df["tweet"]:
        if isinstance(tweet_list, list):
            for tweet in tweet_list:
                if isinstance(tweet, str):
                    words = tweet.lower().split()
                    filtered_words = [w for w in words if w not in stopwords]
                    all_words.extend(filtered_words)
    counter = Counter(all_words)
    most_common_words = counter.most_common(n)
    words, counts = zip(*most_common_words)

    plt.figure(figsize=FIGSIZE)
    bars = plt.bar(words, counts, color=COLOR, alpha=ALPHA)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            f"{yval}",
            ha="center",
            va="bottom",
            fontsize=BAR_FONTSIZE,
        )

    plt.title(
        f"Top {n} Most Common Words (without stop words) for label {label}",
        fontsize=TITLE_FONTSIZE,
    )
    plt.xlabel("Words", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Frequency", fontsize=LABEL_FONTSIZE)
    plt.xticks(rotation=ROTATION, fontsize=TICKS_FONTSIZE)
    plt.show()

def plot_column_vs_label(df: pd.DataFrame, column: str) -> None:
    """
    Plots the distribution of bot and human labels within any boolean column (e.g., verified, default_profile).

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the column and 'label'.
        column (str): Name of the boolean column to group by.
    """

    label_map = {0: 'Human', 1: 'Bot'}
    value_map = {True: 'Yes', False: 'No'}

    df_copy = df.copy()
    df_copy['label_text'] = df_copy['label'].map(label_map)

    df_copy[column] = df_copy[column].map({'True ': True, 'False ': False})
    df_copy['column_text'] = df_copy[column].map(value_map)

    _, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(
        data=df_copy,
        x='column_text',
        hue='label_text',
        multiple='fill',
        shrink=0.8,
        palette=PALETTE,
        edgecolor='black',
        alpha=ALPHA,
        ax=ax
    )

    ax.set_ylabel('Proportion')
    ax.set_xlabel(column.replace('_', ' ').capitalize())
    ax.set_title(f'Label Distribution by {column.replace("_", " ").capitalize()}', fontsize=TITLE_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.tight_layout()
    plt.show()