# Prompting the Mistrel LLM on the Feedback Prize Datasets
# Created by Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path
from vllm import (LLM,
                  SamplingParams, )

import argparse

import pandas as pd


def main(args: argparse.Namespace):

    train_df = pd.read_csv(args.data[0])

    print(train_df.head())

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model=args.models[0])

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Data required.\n \n",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=Path,
        nargs="+",
        default=["facebook/opt-125m"],  # Will not treat it as a 1-element array otherwise
        help="Data required.\n \n",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="linear_fitter.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run prompt-based LLMs via vllm.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
