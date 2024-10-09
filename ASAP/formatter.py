# Formats data for the model
# Alejandro Ciuba, alc307@pitt.edu
import pandas as pd


def main():

    df = pd.read_csv("init/asap-aes/training_set_rel3.tsv", delimiter="\t", encoding='latin')

    df = df[["essay_id", "essay_set", "essay", "domain1_score"]].groupby(["essay_set"]).sample(n=144)

    for prompt in range(1, 8):

        prompt_df = df[df["essay_set"] == prompt]
        prompt_df = prompt_df[["essay_id", "essay", "domain1_score"]]
        prompt_df.to_csv(f"data/p{prompt}_test.txt", sep="\t", header=False, index=False)

if __name__ == "__main__":
    main()
