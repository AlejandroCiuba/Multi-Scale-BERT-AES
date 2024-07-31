# Finetune the model on a different prompt
# Created by Alejandro Ciuba, alc307@pitt.edu
# Ask about how to implement the loss function
# Ask if we should retrain/train from scratch/finetune
# Ask about how to train/finetune through HuggingFace
from ASAPDataset import (ASAPDataset,
                         ToEncoded, )
from model_architechure_bert_multi_scale_multi_loss import DocumentBertScoringModel
from pathlib import Path
from torch.utils.data import (DataLoader,
                              Dataset, )
from transformers import BertTokenizer

import argparse
import torch

import pandas as pd


def load_dataset(data: Path, prompt: int,
                 chunk_sizes, tokenizer: BertTokenizer) -> DataLoader:

    df = pd.read_csv(data, index_col=0)
    df['split'] = df['split'].astype(int)

    dataset = ASAPDataset(
        data=df,
        train_splits=list(range(0,3)),
        valid_split=3,
        test_split=4,
        prompt=prompt[0],
        transform=ToEncoded(tokenizer=tokenizer,
                            chunk_sizes=chunk_sizes),
    )

    return dataset


def main(args: argparse.Namespace):

    # Have to do this because of how DocumentBertScoringModel parses the args
    args.prompt = args.prompt * 2

    arch_model = DocumentBertScoringModel(args=args)
    dataloader = load_dataset(
        data=args.data,
        prompt=args.prompt,
        chunk_sizes=arch_model.chunk_sizes,
        tokenizer=arch_model.bert_tokenizer,
    )


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Data.\n \n",
    )

    parser.add_argument(
        "-m",
        "--bert_model_path",
        type=str,
        required=True,
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
        "-p",
        "--prompt",
        type=int,
        nargs="+",
        default=[3],
        help="Prompt to train the model on.\n \n",
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
        type=str,
        default=".",
        help="file to store the final results in.\n \n",
    )

    parser.add_argument(
        "-v",
        "--device",
        type=str,
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
