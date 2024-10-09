from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support, )
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import argparse

import numpy as np

def main(args: argparse.Namespace):

    with open(args.data) as src:
        y, X = list(zip(*[(float(line.split()[0]), float(line.split()[1])) for line in src]))

    X, y = np.array(list(X)).reshape(-1, 1), np.array(list(y))

    model = make_pipeline(StandardScaler(),
                          LinearRegression(), )

    print(model.steps)

    model.fit(X, y)

    y_pred = model.predict(X)

    print(accuracy_score(y, y_pred.round()))
    print(precision_recall_fscore_support(y, y_pred.round()))

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Data required.\n \n",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="linear_fitter.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Linear fitter for the prompt 8 model.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
