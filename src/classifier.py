#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : spam classify evaluation for exp1
# * Last change   : 20:47:53 2020-03-24
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

from sklearn.utils import shuffle
from loguru import logger
from loguru._defaults import LOGURU_FORMAT
from joblib import Parallel, delayed
from itertools import product
import os.path as ospath
import pandas as pd
import numpy as np
import sys
import time
import json


@click.command()
@click.option("--dataset", "-d", default="trec06p", help="name of dataset")
@click.option("--n-workers", "-n", default=1, help="num of processes")
@click.option("--n-fold", "-f", default=5, help="num of folds in cross validation")
@click.option("--train-ratio", "-r", default=1.0, help="ratio of training dataset to use")
@click.option("--random-seed", "-s", default=5, help="random seed for data shuffling")
@click.option("--output-dir", "-o", default="./result", help="output directory")
@click.option("--model", "-m", default="naive_bayes_0", help="model name for training")
@click.option("--use-meta", "-m", default=True, help="use meta data")
@click.option("--all-test", "-a", is_flag=True, help="run all possible tests")
def main(dataset, output_dir, **kwargs):
    logger.remove()
    logger.add(
        sys.stdout, level="INFO",
        format="[<green>{time}</green>, <blue>{level}</blue> <white>{message}</white>]"
    )
    script_dir = Path(ospath.dirname(ospath.abspath(__file__)))
    input_dir = script_dir.parent / "data" / dataset / "preprocess"
    output_dir = Path(ospath.abspath(output_dir))
    n_fold_cross_validation(input_dir, output_dir, **kwargs)


def n_fold_cross_validation(input_dir, output_dir, n_fold=5, n_workers=1, **kwargs):
    logger.info(f"{n_fold} fold cross validation, train ratio {kwargs['train_ratio']}, random seed {kwargs['random_seed']}")
    df_label = shuffle(pd.read_csv(input_dir / "label.csv", dtype=str), random_state=kwargs["random_seed"])
    df_label.label = df_label.label.astype(int)
    n_data = df_label.shape[0]
    train_masks = [(i*n_data//n_fold, (i+1)*n_data//n_fold) for i in range(n_fold)]
    options = {
        "n_job": n_workers,
        "backend": "multiprocessing",
        "verbose": 100,
    }
    if kwargs.pop("all_test", False): # run for all tests
        list_train_ratio = [0.05, 0.50, 1.00]
        list_use_meta = [True, False]
        list_model = [f"naive_bayes_{i}" for i in range(5)]
        tasks = product(train_masks, list_train_ratio, list_model, list_use_meta)
        results = Parallel(**options)(delayed(train_and_evaluate)(df_label, input_dir, *opt) for opt in tasks)
    else:
        results = Parallel(**options)(delayed(train_and_evaluate)(df_label, input_dir, m, **kwargs) for m in train_masks)

    df_result = pd.DataFrame.from_records(results)
    df_result.to_csv(output_dir / "result_{time.strftime('%x').replace('/', '_')}.csv", index=False)

    logger.info("\n"+ str(df_result))
    logger.info(f"mean measure:\n {df_result.mean()}")


def train_and_evaluate(df_label, input_dir, train_mask, train_ratio, model, use_meta, **kwargs):
    t0 = time.time()

    logger.remove()
    logger.add(
        sys.stderr, level="DEBUG",
        format=f"<yellow>{model}_{train_mask}_{train_ratio}</yellow> - {LOGURU_FORMAT}",
        backtrace=True
    )

    n_data = df_label.shape[0]
    start, end = train_mask
    mask = np.ones(n_data).astype(bool)
    mask[start:end] = False
    df_label_train = df_label[mask]
    df_label_train = df_label_train[:df_label_train.shape[0]*train_ratio]
    df_label_test = df_label[~mask]
    
    logger.debug(f"train: {df_label_train.shape[0]}, {np.sum(df_label_train.label)} of spam")
    logger.debug(f"test: {df_label_test.shape[0]}")

    # train ------------------------------
    prob_p = df_label_train.label.values.sum() / df_label_train.shape[0]
    logger.debug(f"P(p): {prob_p:.4f}")

    prob_p_meta, prob_n_meta = get_meta_prob(df_label_train, input_dir)[model]
    prob_p_text, prob_n_text = get_text_prob(df_label_train, input_dir)[model]

    t1 = time.time()

    # test -------------------------------
    def predict(filename):
        with open(input_dir / f"{filename}.txt", "r") as f:
            words = zip(*np.unique(f.read().split(), return_counts=True))
        _prob_p = np.prod(list(map(prob_p_text, words))) * prob_p
        _prob_n = np.prod(list(map(prob_n_text, words))) * (1-prob_p)
        if use_meta:
            with open(input_dir / f"{filename}.meta", "r") as f:
                metas = [(k, len(v)) for k, v in json.load(f).items()]
            _prob_p *= np.prod(list(map(prob_p_meta, metas)))
            _prob_n *= np.prod(list(map(prob_n_meta, metas)))
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
    result["model"] = model
    result["train_mask"] = train_mask
    result["train_ratio"] = train_ratio
    result["use_meta"] = use_meta

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


def get_prob_bayes(_df, _k, _l, _r, _clip=1, alpha=1, M=2): # NOTE parameters
    if _k not in _df.columns:
        ret = alpha / (_df.shape[0] - 1 + M*alpha)
    else:
        if _df[_k].iloc[-1] == 0:
            ret = _df[_k].iloc[-1].values[np.logical_and(
                _df[_k].iloc[-1] >= _l,
                _df[_k].iloc[-1] < _r
            )].clip(max=_clip).sum()
            _df[_k].iloc[-1] = ret
        else:
            ret = _df[_k].iloc[-1]
    return ret


def prob_wrapper(df_p, df_n):
    return {
        "naive_bayes_0": ( # probability based on "exsistence"
            lambda x, _: get_prob_bayes(df_p, x, _l=1, _r=float("inf"), _clip=1),
            lambda x, _: get_prob_bayes(df_n, x, _l=1, _r=float("inf"), _clip=1)
        ),
        "naive_bayes_1": ( # probability based on "exsistence on each time"
            lambda k, v: get_prob_bayes(df_p, x, _l=1, _r=float("inf"), _clip=1) ** v,
            lambda k, v: get_prob_bayes(df_n, x, _l=1, _r=float("inf"), _clip=1) ** v
        ),
        "naive_bayes_2": ( # probability based on "greater or equal than"
            lambda k, v: get_prob_bayes(df_p, k, _l=v, _r=float("inf"), _clip=1),
            lambda k, v: get_prob_bayes(df_n, k, _l=v, _r=float("inf"), _clip=1)
        ),
        "naive_bayes_3": ( # probability based on "accurate value"
            lambda k, v: get_prob_bayes(df_p, k, _v=v, _r=v, _clip=1),
            lambda k, v: get_prob_bayes(df_n, k, _v=v, _r=v, _clip=1)
        ),
        "naive_bayes_4": ( # weight based on "exsistence"
            lambda x, _: get_prob_bayes(df_p, x, _l=1, _r=float("inf"), _clip=float("inf")),
            lambda x, _: get_prob_bayes(df_n, x, _l=1, _r=float("inf"), _clip=float("inf"))
        ),
    }


def get_meta_prob(df_label_train, input_dir):
    records = []
    for filename in df_label_train.file:
        with open(input_dir / f"{filename}.meta", "r") as f:
            meta_obj = json.load(f)
        records.append(dict(map(lambda x: (x[0], len[x[1]]), meta_obj)))

    df_meta = pd.DataFrame.from_records(records)
    df_meta_p = df_model.loc[df_label_train.label == 1]
    df_meta_n = df_model.loc[df_label_train.label == 0]
    
    logger.debug(f"train meta data: {len(df_meta.columns)}")

    # append one row as probability (only calculated when needed)
    zero_record = dict(zip(df_meta.columns, np.zeros(df_meta.shape[1])))
    df_meta_p = df_meta_p.append(zero_record).fillna(0)
    df_meta_n = df_meta_n.append(zero_record).fillna(0)

    return prob_wrapper(df_meta_p, df_meta_n)


def get_text_prob(df_label_train, input_dir):
    records = []
    for filename in df_label_train.file:
        with open(input_dir / f"{filename}.txt", "r") as f:
            words = np.array([f.read().split()])
        records.append(dict(zip(*np.unique(words, return_counts=True))))

    df_model = pd.DataFrame.from_records(records)
    df_model_p = df_model.loc[df_label_train.label == 1]
    df_model_n = df_model.loc[df_label_train.label == 0]

    logger.debug(f"train words: {len(df_model.columns)}")

    # append one row as probability (only calculated when needed)
    zero_record = dict(zip(df_model.columns, np.zeros(df_model.shape[1])))
    df_model_p = df_model_p.append(zero_record).fillna(0)
    df_model_n = df_model_n.append(zero_record).fillna(0)

    return prob_wrapper(df_model_p, df_model_n)


if __name__ == "main":
    main()
