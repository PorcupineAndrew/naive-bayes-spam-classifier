#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : data preprocess for exp1
# * Last change   : 09:30:21 2020-03-27
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

from pathlib import Path
from glob import glob
from joblib import Parallel, delayed
import pandas as pd
import os
import click
import shutil
import json
import re

@click.command()
@click.option("--dataset", "-d", default="trec06p", help="name of dataset")
@click.option("--num-workers", "-n", default=1, help="num of processes")
@click.option("--reprocess", "-r", is_flag=True, help="delete existing results")
def main(dataset, num_workers, **kwargs):
    # path
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_dir = script_dir.parent / "data" / dataset
    output_dir = data_dir / "preprocess"
    createDir(output_dir, kwargs.pop("reprocess", False))

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

    # files
    files = list(map(Path, glob(str(data_dir / "data" / "*" / "*"))))
    assert len(files) == df.file.size

    # run
    options = {
        "n_jobs": num_workers,
        "backend": "multiprocessing",
        "verbose": 100,
    }
    results = Parallel(**options)(delayed(executor)(f, output_dir) for f in files)
    results = [i for i in results if i]
    print(f"unsuccessful process: {len(results)}")
    print(f"successful process  : {df.shape[0]-len(results)}")

    df.set_index("file").drop(results).reset_index().to_csv(output_dir / "label.csv", index=False)


def executor(input_path, output_dir):
    try:
        with open(input_path, "r") as f:
            paras = re.split(r"\n\n", f.read()) # get paragraphs
    except UnicodeDecodeError as e:
        # return file id
        return f"{input_path.parent.stem}{input_path.stem}"

    # save meta data
    meta_file = output_dir / f"{input_path.parent.stem}{input_path.stem}.meta"
    meta_data = paras.pop(0)
    try:
        with open(meta_file, "w+") as f:
            json.dump(meta_parse(meta_data), f, indent=4)
    except AttributeError as e:
        # shouldn not happen
        return f"{input_path.parent.stem}{input_path.stem}"

    # save clean text word
    text_file = output_dir / f"{input_path.parent.stem}{input_path.stem}.txt"
    text_data = " ".join(map(text_clean, paras))
    with open(text_file, "w+") as f:
        f.write(text_data)


def createDir(path, remove=False):
    if os.path.exists(path) and remove:
        shutil.rmtree(path)
    if not os.path.exists(path): os.makedirs(path)


def meta_parse(line):
    lines = re.sub(r"\n\s+", " ", line).split("\n") # merge multi-lines meta data
    meta_data = [re.match(r"^(.*?): (.*)$", i.strip()) for i in lines]
    meta_data = [i.groups() for i in meta_data if i]
    ret = {}
    for k, v in meta_data:
        if len(k) > 32: continue
        if k in ret: ret[k].append(v)
        else: ret[k] = [v]
    return ret


def text_clean(line):
    return " ".join([i.strip() for i in re.split(r"[^\w]+", line) if i])


if __name__ == "__main__":
    main()
