# Get the results of the finetune tests
# Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(df: pd.DataFrame, col:str, title: str, save_to: str):

    ax = sns.barplot(data=df, x="Prompt", y=col)
    ax.set_title(title)

    for i in ax.containers:
        ax.bar_label(i, fmt="%.3f")

    plt.savefig(f"plots/{save_to}.pdf", format="pdf")
    plt.close()


def get_data(path: Path):

    pearsons, qwks, prompts, losses = [], [], [], []
    for file in path.glob("*.out"):

            ps, qs = [], []
            prompt = ""

            with open(file, 'r+') as src:

                for line in src:

                    if 'pearson' in line:
                        ps.append(line)

                    elif 'qwk' in line:
                        qs.append(line)

                    elif 'Losses:' in line:
                        losses.append((prompt, [float(num) for num in line[8:].split(',')]))  # Relies on the prompt line being before the loss list

                    elif "prompt:" in line:
                        prompt = line[7]

            pearsons.append(float(ps[-1][8:]))
            qwks.append(float(qs[-1][4:]))
            prompts.append(prompt)

    return pd.DataFrame({"Prompt": prompts,
                         "Pearson": pearsons,
                         "QWK": qwks, }).sort_values(by="Prompt"), losses


def main():

    plt.style.use('ggplot')

    after = Path('init/finetune_results/')
    before = Path('init/no-fix_results/')

    before_df, _ = get_data(before)
    after_df, after_losses = get_data(after)

    plot(before_df, "Pearson","Pearson per Prompt before Finetuning", "pearson-before")
    plot(before_df, "QWK", "QWK per Prompt before Finetuning", "qwk-before")

    plot(after_df, "Pearson","Pearson per Prompt after Finetuning", "pearson-after")
    plot(after_df, "QWK", "QWK per Prompt after Finetuning", "qwk-after")

    print(after_df.Pearson.mean())
    print(after_df.QWK.mean())

    for prompt, loss in after_losses:
        plt.plot(loss)
        plt.title(label=f"{prompt} Loss")
        plt.show()


if __name__ == "__main__":
    main()
