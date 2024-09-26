#! usr/bin/bash

DIRECTORY=data/feedback-prize-english-language-learning/train.csv

python prompt-llms.py -d $DIRECTORY -m models
