# Finetune the model on a different prompt
# Created by Alejandro Ciuba, alc307@pitt.edu
# Ask about how to implement the loss function
# Ask if we should retrain/train from scratch/finetune
# Ask about how to train/finetune through HuggingFace
from ASAPDataset import ASAPDataset
from model_architechure_bert_multi_scale_multi_loss import DocumentBertScoringModel
from pathlib import Path
from torch.utils.data import (DataLoader,
                              Dataset, )

import argparse
import torch

import pandas as pd


def main(args: argparse.Namespace):

    df = pd.read_csv("REPLACE", index_col=0)
    df['split'] = df['split'].astype(int)

    print(df.sample(1))

    dataset = ASAPDataset(data=df, train_splits=list(range(0,3)), valid_split=3, test_split=4, prompt=3)

    print(dataset.X_train[:2])
    print(dataset.y_train[:2])


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Data.\n \n",
    )

    parser.add_argument(
        "-m",
        "--bert_model_path",
        type=Path,
        default=Path,
        help="Model.\n \n",
    )

    parser.add_argument(
        "-c",
        "--chunk_sizes",
        type=str,
        default="90_30_130_10",
        help="Chunk sizes for the segmented model.\n \n",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Number of datapoints per batch.\n \n",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to finetune the model.\n \n",
    )

    parser.add_argument(
        "-r",
        "--result_file",
        type=Path,
        default=Path("."),
        help="file to store the final results in.\n \n",
    )

    parser.add_argument(
        "-v",
        "--device",
        type="str",
        default="cuda",
        help="Device to run the training on.\n \n",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="finetune.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Finetune the BERT model similarly to the paper.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
