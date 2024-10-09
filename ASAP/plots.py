from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():

    plt.style.use('ggplot')

    path = Path("results/fix2-plots-corr/")
    corrs = {"name": [],
             "correlation": [], }

    for file in path.glob("*.txt"):

        with open(file, 'r') as src:

            corrs["name"].append(f"8-to-{file.name[9]}")
            corrs["correlation"].append(float(src.readlines(-1)[1][29:46]))  # Might have to change depending on the fix

    corrs_df = pd.DataFrame(data=corrs).sort_values(by="correlation")

    ax = sns.barplot(data=corrs_df, x="name", y="correlation")

    ax.set_title("Model Output to Gold Label Spearman Correlation\nAfter Binning\nn=144", size=10)
    ax.set_xlabel("Model-to-Prompt")
    ax.set_ylabel("Spearman Correlation")

    plt.savefig(f"results/spearman-plots/fix2.png")
    # plt.show()


if __name__ == "__main__":
    main()
