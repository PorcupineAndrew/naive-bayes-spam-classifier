#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : spam classify evaluation for exp1
# * Last change   : 00:50:04 2020-03-21
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

from sklearn.utils import shuffle
from loguru import logger
from loguru._defaults import LOGURU_FORMAT
from joblib import Parallel, delayed
import os.path as ospath
import pandas as pd
import numpy as np
import sys
import time


@click.command()
@click.option("--dataset", "-d", default="trec06p", help="name of dataset")
@click.option("--num-workers", "-n", default=1, help="num of processes")
@click.option("--num-fold", "-f", default=5, help="num of folds in cross validation")
@click.option("--train_ratio", "-r", default=1.0, help="ratio of training dataset to use")
def main(dataset, num_workers, num_fold, **kwargs):
    logger.remove()
    logger.add(
        sys.stdout, level="INFO",
        format="[<green>{time}</green>, <blue>{level}</blue> <white>{message}</white>]"
    )
    script_dir = Path(ospath.dirname(ospath.abspath(__file__)))
    input_dir = script_dir.parent / "data" / dataset / "preprocess"
    n_fold_cross_validation(input_dir, num_workers, train_ratio)


def n_fold_cross_validation(input_dir, n_fold=5, train_ratio=1.0, **kwargs):
    logger.info(f"{n_fold} fold cross validation, train ratio {train_ratio}")
    df_label = shuffle(pd.read_csv(input_dir / "label.csv", dtype=str))
    df_label.label = df_label.label.astype(int)
    n_data = df_label.shape[0]
    train_masks = [(i*n_data//n_fold, (i+1)*n_data//n_fold) for i in range(n_fold)]
    options = {
        "n_job": num_workers,
        "backend": "multiprocessing",
        "verbose": 100,
    }
    results = Parallel(**options)(delayed(train_and_evaluate)(df_label, m, input_dir, train_ratio, **kwargs) for m in train_masks)
    logger.info("\n"+ str(results))


def train_and_evaluate(df_label, train_mask, input_dir, train_ratio, **kwargs):
    st_time = time.time()

    logger.remove()
    logger.add(
        sys.stderr, level="DEBUG",
        format=f"<yellow>{train_mask}</yellow> - {LOGURU_FORMAT}",
        backtrace=True
    )

    n_data = df_label.shape[0]
    start, end = train_mask
    mask = np.ones(n_data).astype(bool)
    mask[start:end] = False
    df_label_train = df_label[mask]
    df_label_train = df_label_train[:df_label_train.shape[0]*train_ratio]
    df_label_test = df_label[~mask]
    
    logger.debug(f"train: {df_label_train.shape[0]}, {np.sum(df_label_train.label)} of spam, train ratio {train_ratio}")
    logger.debug(f"test: {df_label_test.shape[0]}")

    # train ------------------------------
    # TODO with meta data
    records = []
    for filename in df_label_train.file:
        with open(input_dir / f"{filename}.txt", "r") as f:
            words = np.array([f.read().split()])
        records.append(dict(zip(*np.unique(words, return_counts=True))))

    df_model = pd.DataFrame.from_records(records)
    df_model_p = df_model.loc[df_label_train.label == 1]
    df_model_n = df_model.loc[df_label_train.label == 0]
    prob_p = df_model_p.size / df_model.size

    logger.debug(f"train words: {len(df_model.columns)}")
    logger.debug(f"P(p): {prob_p:.4f}")

    # append one row as probability (only calculated when needed)
    zero_record = dict(zip(df_model.columns, np.zeros(df_model.shape[1])))
    df_model_p = df_model_p.append(zero_record).fillna(0)
    df_model_n = df_model_n.append(zero_record).fillna(0)

    def get_prob(_df, _word, clip=1, alpha=1, M=2): # NOTE parameters
        if _word not in _df.columns:
            ret = alpha / (_df.shape[0] - 1 + M*alpha)
        else:
            if _df[_word].iloc[-1] == 0:
                ret = (np.sum(_df[_word].iloc[:-1].clip_upper(clip)) + alpha) \
                        / (_df.shape[0] - 1 + M*alpha)
                _df[_word].iloc[-1] = ret
            else:
                ret = _df[_word].iloc[-1]
        return ret
    get_prob_p = lambda x: get_prob(df_model_p, x)
    get_prob_n = lambda x: get_prob(df_model_n, x)

    t1 = time.time()

    # test -------------------------------
    def predict(filename):
        with open(input_dir / f"{filename}.txt", "r") as f:
            words = np.array([f.read().split()])
        _prob_p = np.prod(list(map(get_prob_p, words))) * prob_p
        _prob_n = np.prod(list(map(get_prob_n, words))) * (1-prob_p)
        return int(_prob_p > _prob_n)

    predicts = np.array(list(map(predict, df_label_test.file)))
    labels = df_label_test.label.values

    tp = sum(predict & labels)
    fp = sum(predict & ~labels)
    fn = sum(~predict & labels)
    tn = sum(~predict & ~labels)
    assert tp + fp + fn + tn == len(predict)

    t2 = time.time()

    result = {}
    result["accuracy"] = (tp + tn) / (tp + fp + fn + tn)
    result["precision"] = tp / (tp + fp)
    result["recall"] = tp / (tp + fn)
    result["specificity"] = tn / (tn + fp)
    result["prevalence"] = (tp + fn) / (tp + fp + fn + tn)
    result["f1-score"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
    result["train_time"] = t1 - st_time
    result["test_time"] = t2 - t1

    for k,v in result.items():
        logger.debug(f"{k:15s}: {v:.4f}")
    return result
