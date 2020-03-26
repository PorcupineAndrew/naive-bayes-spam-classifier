#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : spam classify evaluation for exp1
# * Last change   : 10:42:44 2020-03-27
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

from sklearn.utils import shuffle
from loguru import logger
from loguru._defaults import LOGURU_FORMAT
from joblib import Parallel, delayed
from itertools import product
from pathlib import Path
from tqdm import tqdm
import os.path as ospath
import pandas as pd
import numpy as np
import click
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
    kwargs.update({"tqdm_disable": n_workers>1})
    logger.info(f"{n_fold} fold cross validation, train ratio {kwargs['train_ratio']}, random seed {kwargs['random_seed']}")
    df_label = shuffle(pd.read_csv(input_dir / "label.csv", dtype=str), random_state=kwargs["random_seed"])
    df_label.label = df_label.label.astype(int)
    n_data = df_label.shape[0]
    train_masks = [(i*n_data//n_fold, (i+1)*n_data//n_fold) for i in range(n_fold)]
    options = {
        "n_jobs": n_workers,
        "backend": "multiprocessing",
        "verbose": 100,
    }
    if kwargs.pop("all_test", False): # run for all tests
        list_train_ratio = [0.05, 0.50, 1.00]
        list_use_meta = [True, False]
        list_model = [f"naive_bayes_{i}" for i in range(5)]
        tasks = product(train_masks, list_train_ratio, list_model, list_use_meta)
        for i in ["train_ratio", "use_meta", "model"]: kwargs.pop(i)
        results = Parallel(**options)(delayed(train_and_evaluate)(df_label, input_dir, *opt, **kwargs) for opt in shuffle(list(tasks)))
    else:
        results = Parallel(**options)(delayed(train_and_evaluate)(df_label, input_dir, m, **kwargs) for m in train_masks)

    df_result = pd.DataFrame.from_records(results)
    df_result.to_csv(output_dir / f"result_{time.strftime('%x').replace('/', '_')}.csv", index=False)

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
    df_label_train = df_label_train[:int(df_label_train.shape[0]*train_ratio)]
    df_label_test = df_label[~mask]
    
    logger.debug(f"train: {df_label_train.shape[0]}, {np.sum(df_label_train.label)} of spam")
    logger.debug(f"test: {df_label_test.shape[0]}")

    # train ------------------------------
    prob_p = df_label_train.label.values.sum() / df_label_train.shape[0]
    prob_n = 1 - prob_p
    logger.debug(f"P(p): {prob_p:.4f}")

    prob_p_meta, prob_n_meta = get_meta_prob(df_label_train, input_dir, **kwargs)[model]
    prob_p_text, prob_n_text = get_text_prob(df_label_train, input_dir, **kwargs)[model]

    logger.debug(f"finish training")
    t1 = time.time()

    # test -------------------------------
    def predict(filename):
        with open(input_dir / f"{filename}.txt", "r") as f:
            words = np.unique(f.read().split(), return_counts=True)
        # NOTE: to avoid overflow
        # _prob_p = np.prod(list(map(prob_p_text, *words))) * prob_p
        # _prob_n = np.prod(list(map(prob_n_text, *words))) * (1-prob_p)
        _prob_p = np.log(prob_p) + np.sum(list(map(prob_p_text, *words)))
        _prob_n = np.log(prob_n) + np.sum(list(map(prob_n_text, *words)))
        if use_meta:
            with open(input_dir / f"{filename}.meta", "r") as f:
                meta_obj = json.load(f)
            metas = meta_obj.keys(), map(lambda x: len(x), meta_obj.values())
            # _prob_p *= np.prod(list(map(prob_p_meta, *metas)))
            # _prob_n *= np.prod(list(map(prob_n_meta, *metas)))
            _prob_p += np.sum(list(map(prob_p_meta, *metas)))
            _prob_n += np.sum(list(map(prob_n_meta, *metas)))
        return int(_prob_p > _prob_n)

    files = tqdm(df_label_test.file, disable=kwargs["tqdm_disable"])
    files.set_description("Testing")
    predicts = np.array(list(map(predict, files))).astype(bool)
    labels = df_label_test.label.values.astype(bool)

    tp = sum(predicts & labels)
    fp = sum(predicts & ~labels)
    fn = sum(~predicts & labels)
    tn = sum(~predicts & ~labels)
    assert tp + fp + fn + tn == len(predicts), f'{tp} + {fp} + {fn} + {tn} != {len(predicts)}'
    logger.debug(f'{tp}, {fp}, {fn}, {tn}')

    t2 = time.time()

    result = {}
    result["model"] = model
    result["train_mask"] = train_mask
    result["train_ratio"] = train_ratio
    result["use_meta"] = use_meta

    with np.errstate(divide="ignore", invalid="ignore"):
        result["accuracy"] = (tp + tn) / (tp + fp + fn + tn)
        result["precision"] = tp / (tp + fp)
        result["recall"] = tp / (tp + fn)
        result["specificity"] = tn / (tn + fp)
        result["prevalence"] = (tp + fn) / (tp + fp + fn + tn)
        result["f1-score"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
        result["train_time"] = t1 - t0
        result["test_time"] = t2 - t1

    for k,v in result.items():
        logger.debug(f"{k:15s}: {v}")
    return result


def get_prob_bayes(_data, _k, _l, _r, _clip=1, alpha=1, M=2, post=np.log): # NOTE parameters
    density = 0
    if _k in _data:
        counts, freq = np.array(list(_data[_k].items())).T
        valid = (counts >= _l) & (counts < _r)
        density = (freq[valid] * counts[valid].clip(max=_clip)).sum()
    return post((density + alpha) / (_data["__total__"] + M*alpha))


def prob_wrapper(data_p, data_n):
    return {
        "naive_bayes_0": ( # probability based on "exsistence"
            lambda x, _: get_prob_bayes(data_p, x, _l=1, _r=float("inf"), _clip=1),
            lambda x, _: get_prob_bayes(data_n, x, _l=1, _r=float("inf"), _clip=1)
        ),
        "naive_bayes_1": ( # probability based on "exsistence on each time"
            lambda k, v: get_prob_bayes(data_p, k, _l=1, _r=float("inf"), _clip=1) * v,
            lambda k, v: get_prob_bayes(data_n, k, _l=1, _r=float("inf"), _clip=1) * v
        ),
        "naive_bayes_2": ( # probability based on "greater or equal than"
            lambda k, v: get_prob_bayes(data_p, k, _l=v, _r=float("inf"), _clip=1),
            lambda k, v: get_prob_bayes(data_n, k, _l=v, _r=float("inf"), _clip=1)
        ),
        "naive_bayes_3": ( # probability based on "accurate value"
            lambda k, v: get_prob_bayes(data_p, k, _l=v, _r=v, _clip=1),
            lambda k, v: get_prob_bayes(data_n, k, _l=v, _r=v, _clip=1)
        ),
        "naive_bayes_4": ( # weight based on "exsistence"
            lambda x, _: get_prob_bayes(data_p, x, _l=1, _r=float("inf"), _clip=float("inf")),
            lambda x, _: get_prob_bayes(data_n, x, _l=1, _r=float("inf"), _clip=float("inf"))
        ),
    }


def update(record, k, v):
    record["__total__"] += 1
    if k in record:
        if v in record[k]: record[k][v] += 1
        else: record[k][v] = 1
    else: record[k] = dict([(v, 1)])


def get_meta_prob(df_label_train, input_dir, **kwargs):
    p_records, n_records = dict(__total__=0), dict(__total__=0)

    files = tqdm(np.array(df_label_train), disable=kwargs["tqdm_disable"])
    for filename, label in files:
        files.set_description(f"Loading {filename}.meta")
        with open(input_dir / f"{filename}.meta", "r") as f:
            meta_obj = json.load(f)
        list(map(lambda k, v: update(p_records if label == 1 else n_records, k, v),
            meta_obj.keys(), map(len, meta_obj.values())))

    return prob_wrapper(p_records, n_records)


def get_text_prob(df_label_train, input_dir, **kwargs):
    p_records, n_records = dict(__total__=0), dict(__total__=0)

    files = tqdm(np.array(df_label_train), disable=kwargs["tqdm_disable"])
    for filename, label in files:
        files.set_description(f"Loading {filename}.txt")
        with open(input_dir / f"{filename}.txt", "r") as f:
            words = np.array([f.read().split()])
        list(map(lambda k, v: update(p_records if label == 1 else n_records, k, v),
            *np.unique(words, return_counts=True)))

    return prob_wrapper(p_records, n_records)


if __name__ == "__main__":
    main()
