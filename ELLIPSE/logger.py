# Generic logging setup
# Created by Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path

import logging

StrPath = str | Path

def make_logger(logpath: StrPath, errpath: StrPath, **kwargs) \
    -> tuple[logging.LoggerAdapter, logging.LoggerAdapter]:

    # Set up logger
    # version_tracking = {"version": "VERSION %s" % VERSION}

    fmt = logging.Formatter(fmt="%(version)s : %(asctime)s : %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')

    # Normal logging
    logger = logging.getLogger("data_log")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(logpath)
    handler.setFormatter(fmt)

    logger.addHandler(handler)

    # Error logging
    err = logging.getLogger("err_log")
    logger.setLevel(logging.WARNING)

    handler = logging.FileHandler(errpath)
    handler.setFormatter(fmt)

    err.addHandler(handler)

    logger = logging.LoggerAdapter(logger, kwargs)
    err = logging.LoggerAdapter(err, kwargs)

    return logger, err
