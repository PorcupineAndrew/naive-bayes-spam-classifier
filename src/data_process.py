#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : data preprocess for exp1
# * Last change   : 00:16:00 2020-03-20
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

from pathlib import Path
from glob import glob
from joblib import Parallel, delayed
import os.path as ospath
import pandas as pd
import click
import shutil
import json
import re

@click.command()
@click.option("--dataset", "-d", default="trec06p", help="name of dataset")
@click.option("--num-workers", "-n", default=1, help="num of processes")
@click.option("--reprocess", "-r", default=False, help="delete existing results")
def main(dataset, num_workers, reprocess, **kwargs):
    # path
    script_dir = Path(ospath.dirname(ospath.abspath(__file__)))
    data_dir = script_dir.parent / "data" / dataset
    output_dir = data_dir / "preprocess"
    createDir(output_dir, reprocess)

    # label
    options = {
        "dtype": str,
        "sep": " ",
        "names": ["label", "file"],
        "engine": "c",
    }
    df = pd.read_csv(data_dir / "label" / "index", **options)
    df.label = (df.label == "spam").astype(int)
    df.file = list(map(lambda x: "".join(x.split("/")[-2:]), df.file))
    df.to_csv(output_dir / "label.csv", index=False)

    # files
    files = glob(str(data_dir / "data" / "*" / "*"))
    assert len(files) == df.file.size

    # run
    options = {
        "n_job": num_workers,
        "backend": "multiprocessing",
        "verbose": 100,
    }
    Parallel(**options)(delayed(executor)(f, output_dir) for f in files)


def executor(input_path, output_dir):
    with open(input_path, "r") as f:
        paras = re.split(r"\n\n", f.read())

    # save meta data
    meta_file = output_dir / f"{input_path.parent.stem}{input_path.stem}.meta"
    meta_data = paras.pop(0)
    with open(meta_file, "w+") as f:
        json.dump(meta_parse(meta_data), f, indent=4)

    # save clean text word
    text_file = output_dir / f"{input_path.parent.stem}{input_path.stem}.txt"
    text_data = " ".join(map(text_clean, paras))
    with open(text_file, "w+") as f:
        f.write(text_data)


def createDir(path, remove=False):
    if ospath.exists(path) and remove:
        shutil.rmtree(path)
    os.makedirs(path)


def meta_parse(line):
    lines = re.sub(r"\n\s+", " ", line).split("\n") # merge multi-lines meta data
    meta_data = [re.match(r"^(.*): (.*)$", i.strip()).groups() for i in lines]
    ret = {}
    for k, v in meta_data:
        if k in ret: ret[k].append(v)
        else: ret[k] = [v]
    return ret


def text_clean(line):
    r = re.compile(r"^[^\w]+|[^\w]+$")
    return " ".join([r.sub("", i) for i in line.split()])


if __name__ == "__main__":
    main()
