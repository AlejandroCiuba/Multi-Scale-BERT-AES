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
import datetime
import logger
import logging
import os
import torch

import pandas as pd
import torch.nn as nn


log = logging.getLogger()


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
        valid_split=4,
        test_split=3,
        prompt=prompt[0],
        transform=ToEncoded(tokenizer=tokenizer,
                            chunk_sizes=chunk_sizes),
    )

    return dataset


@torch.no_grad()
def evaluate(model:DocumentBertScoringModel, dataset: ASAPDataset, criterion: ASAPLoss):

        model.eval()
        model.bert_regression_by_word_document.eval()
        model.bert_regression_by_chunk.eval()

        X, y = dataset.get_valid(transform=True)
        X, y = [x.to(args.device) for x in X], y.to(args.device, dtype=torch.float32)

        predictions = model(X)
        loss = criterion(predictions=predictions, targets=y)

        pearson, qwk = model.predict_for_regress(data=dataset.get_valid(transform=False))

        return loss.item(), pearson, qwk


def save_model(model:DocumentBertScoringModel, save_path: str, name: str):

    BertConfig.save_pretrained(model.config, save_directory=save_path)

    chunk_save_path, word_save_path = save_path + f"chunk", save_path + f"word_document"
    os.makedirs(chunk_save_path, exist_ok=True)
    os.makedirs(word_save_path, exist_ok=True)

    torch.save(model.bert_regression_by_chunk.state_dict(), chunk_save_path + f"/{name}")
    torch.save(model.bert_regression_by_word_document.state_dict(), word_save_path + f"/{name}")


def main(args: argparse.Namespace):

    log.info(f"===================== {datetime.datetime.now().isoformat()} =====================")
    log.info(f"===================== FINE-TUNING ON PROMPT {args.prompt[0]} =====================")
    log.info(f"Settings: {args.__str__()}")

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

    model.bert_regression_by_word_document = nn.DataParallel(model.bert_regression_by_word_document)
    model.bert_regression_by_chunk = nn.DataParallel(model.bert_regression_by_chunk)
    model.to(args.device)

    model.float()
    model.bert_regression_by_word_document.float()
    model.bert_regression_by_chunk.float()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = ASAPLoss(dim=0, device=args.device)

    name = args.save_model.split("/")[-1]
    path = args.save_model[:len(args.save_model) - len(name)]
    batches_per_epoch = (len(dataset) // args.batch_size) + 1

    log.info(f"Train/Eval/Test: {len(dataset)}/{dataset.len_split('valid')}/{dataset.len_split('test')}")

    log.info(f"Started training loop for {name}")

    prev_best = 1_000_000
    loss_tracker = []

    best_model = {"DocumentBertScoringModel": None,
                  "bert_regression_by_word_document": None,
                  "bert_regression_by_chunk": None, 
                  "count": -1,}

    for epoch in tqdm(range(args.epochs)):

        model.train()
        model.bert_regression_by_word_document.train()
        model.bert_regression_by_chunk.train()

        for i, _ in enumerate(range(0, len(dataset), args.batch_size)):

            optimizer.zero_grad()

            X, y = dataset[i: i + (args.batch_size)]  # List of 5 tensors containing args.batch_size Tensors each, args.batch_size scores
            X, y = [x.to(args.device) for x in X], y.to(args.device)

            # for x in X:
            #     if torch.any(torch.isnan(x)):
            #         print(x)

            predictions = model(X)
            # log.debug(predictions)
            loss = criterion(predictions=predictions, targets=y)

            loss.backward()
            optimizer.step()

            log.info(f"{epoch}/{args.epochs} | {i}/{batches_per_epoch}: {loss.item():0.5f}")
            loss_tracker.append(loss.item())
            # exit()

        eval_loss, eval_pearson, eval_qwk = evaluate(model=model, dataset=dataset, criterion=criterion)
        log.info(f"{epoch}: Evaluation loss | Pearson | qwk : {eval_loss:.5f} | {eval_pearson:.5f} | {eval_qwk:.5f}")

        if eval_loss < prev_best:

            log.info(f"Saving model {name} on epoch {epoch} ({eval_loss:.5f} is the new best loss!)")
            save_model(model=model, save_path=path, name=name)

            prev_best = eval_loss

            # Save the parameters
            best_model["DocumentBertScoringModel"] = model.state_dict()
            best_model["bert_regression_by_word_document"] = model.bert_regression_by_word_document.state_dict()
            best_model["bert_regression_by_chunk"] = model.bert_regression_by_chunk.state_dict()
            best_model["count"] = epoch

    # Reload the best model parameters
    model.load_state_dict(best_model["DocumentBertScoringModel"])
    model.bert_regression_by_word_document.load_state_dict(best_model["bert_regression_by_word_document"])
    model.bert_regression_by_chunk.load_state_dict(best_model["bert_regression_by_chunk"])

    log.info(f"Using best model from epoch {best_model['count']}")
    log.info("Performing final evaluation on the evaluation set")

    pearson, qwk = model.predict_for_regress(data=dataset.get_valid(transform=False))  # The predict_for_regress function transforms the data
    log.info(f"Pearson: {pearson:.5f} | QWK: {qwk:.5f}")

    log.info("Performing final evaluation on the test set")

    pearson, qwk = model.predict_for_regress(data=dataset.get_test(transform=False))  # The predict_for_regress function transforms the data
    log.info(f"Pearson: {pearson:.5f} | QWK: {qwk:.5f}")

    log.info(f"Losses: {', '.join([str(l) for l in loss_tracker])}")
    log.info(f"Fine-tuning complete on prompt {args.prompt[0]}")


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
        help="Chunk sizes for the segmented model; typed this way due to legacy code.\n \n",
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

    parser.add_argument(
        "-k",
        "--log",
        type=Path,
        default="logs/finetune.log",
        help="Logging output filepath.\n \n",
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

    log, = logger.make_loggers(args.log, levels=logging.INFO)

    main(args)
