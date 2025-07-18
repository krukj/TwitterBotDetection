import matplotlib.pyplot as plt
import pandas as pd

COLOR = "dodgerblue"
ALPHA = 0.8
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 16
TICKS_FONTSIZE = 12
FIGSIZE = (10, 6)


def plot_label_distribution(df: pd.DataFrame) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
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
        )

    ax.set_xlabel("Label", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Percentage (%)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Distribution of labels", fontsize=TITLE_FONTSIZE)
    plt.xticks(rotation=45, fontsize=TICKS_FONTSIZE)
    plt.tight_layout()
    plt.show()
