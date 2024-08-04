# Finetune the model on a different prompt
# Created by Alejandro Ciuba, alc307@pitt.edu
# Ask about how to implement the loss function
# Ask if we should retrain/train from scratch/finetune
# Ask about how to train/finetune through HuggingFace
from ASAPDataset import (ASAPDataset,
                         ASAPLoss,
                         ToEncoded, )
from model_architechure_bert_multi_scale_multi_loss import DocumentBertScoringModel
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (BertConfig,
                          BertPreTrainedModel,
                          BertTokenizer, )

import argparse
import torch

import pandas as pd


def load_dataset(data: Path, prompt: int,
                 chunk_sizes, tokenizer: BertTokenizer,
                 sample: int = -1, seed: int = 42) -> ASAPDataset:

    df = pd.read_csv(data, index_col=0)
    df['split'] = df['split'].astype(int)

    if sample > 0:
        df = df[(df.split == 1) & (df.essay_set == prompt[0])].sample(n=sample, random_state=seed)

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


@torch.no_grad()
def evaluate(model:DocumentBertScoringModel, dataset: ASAPDataset, criterion: ASAPLoss) -> float:

        X, y = dataset.get_valid(transform=True)
        X, y = [x.to(args.device) for x in X], y.to(args.device, dtype=torch.float32)

        for x in X:
            print(x.shape)
        predictions = model(X)
        loss = criterion(predictions=predictions, targets=y)

        model.predict_for_regress(data=dataset.get_valid(transform=False))

        return loss.item()

def main(args: argparse.Namespace):

    # Have to do this because of how DocumentBertScoringModel parses the args
    args.prompt = args.prompt * 2

    model = DocumentBertScoringModel(args=args)
    dataset = load_dataset(
        data=args.data,
        prompt=args.prompt,
        chunk_sizes=model.chunk_sizes,
        tokenizer=model.bert_tokenizer,
        sample=args.sample,
    )

    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = ASAPLoss(dim=0)

    print("Training Set Size:", len(dataset))
    batches_per_epoch = (len(dataset) // args.batch_size) + 1

    evaluate(model, dataset, criterion)
    exit()

    print("Started training loop")
    for epoch in tqdm(range(args.epochs)):

        for i, _ in enumerate(range(0, len(dataset), args.batch_size)):

            optimizer.zero_grad()

            X, y = dataset[i: i + (args.batch_size)]  # List of 5 tensors containing args.batch_size Tensors each, args.batch_size scores
            X, y = [x.to(args.device) for x in X], y.to(args.device)

            predictions = model(X)
            loss = criterion(predictions=predictions, targets=y)

            loss.backward()
            optimizer.step()

            print(f"{epoch}/{args.epochs} | {i}/{batches_per_epoch}: {loss:0.5f}")
            break
        
        eval_loss = evaluate(model=model, dataset=dataset, criterion=criterion)
        print(f"Evaluation loss at epoch {epoch}: {eval_loss:.5f}")

    print("Performing evaluation on the test set")

    model.predict_for_regress(data=dataset.get_test(transform=False))  # The predict_for_regress function transforms the data

    print("Saving the model")

    BertConfig.save_pretrained(model.config, save_directory=args.save_model)

    torch.save(model.bert_regression_by_word_document, args.save_model + "/word_document")
    torch.save(model.bert_regression_by_word_document, args.save_model + "/chunk")


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
        "-x",
        "--sample",
        type=int,
        default=-1,
        help="Sample of datapoints to use; good for testing.\n \n",
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=6e-5,
        help="Learning rate for the entire model.\n \n",
    )

    parser.add_argument(
        "-r",
        "--result_file",
        type=str,
        default=".",
        help="file to store the final results in.\n \n",
    )

    parser.add_argument(
        "-s",
        "--save_model",
        type=str,
        required=True,
        help="Directory and file to save the model to.\n \n",
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
