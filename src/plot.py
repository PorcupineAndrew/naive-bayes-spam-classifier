#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : plot script for result
# * Last change   : 00:21:39 2020-03-28
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

from pathlib import Path
from os.path import dirname, abspath
from itertools import product
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

script_dir = Path(abspath(dirname(__file__)))
output_path = script_dir.parent / "result_figure.png"
df = pd.read_csv(script_dir.parent / "result" / "trec06p" / "result_03_27_20.csv")


# get all identifiers
identifiers = ["model", "train_mask", "train_ratio", "use_meta"]
measures = list(set(df.columns) - set(identifiers))
identifiers = dict(zip(identifiers, map(lambda x: np.sort(np.unique(df[x])), identifiers)))
for k,v in identifiers.items(): print(f"{k:12s}{list(v)}")
n_fold = len(identifiers.pop("train_mask"))


# select best combination
objective = "f1-score"
combinations = list(map(lambda x: dict(zip(["model", "train_ratio", "use_meta"], x)),
                product(*identifiers.values())))
best_comb = combinations[np.argmax(list(map(lambda x: df.loc[np.logical_and.reduce(
                [df[k] == v for k,v in x.items()])][objective].mean(), combinations)))]
print(f"best combination on {objective}: {best_comb}")


# draw comparison plot
def compare_plot(axs, benchmark=best_comb, measure="f1-score"):
    axs = list(axs)
    x = [f"fold_{i}" for i in range(n_fold)]
    kwargs = {
        "linewidth": 2,
        "markersize": 12,
        "marker": ".",
        "alpha": 0.8,
    }
    for element in identifiers.keys():
        ax = axs.pop(0)
        ax.set_title(f"{measure} on {element}")
        ax.grid(True)
        min_y, y_map = 1.0, {}
        for value in identifiers[element]:
            is_benchmark = value == benchmark[element]
            y = df.loc[np.logical_and.reduce(
                    [df[k] == v for k,v in dict(benchmark, **{element: value}).items()])] \
                    .sort_values(by=["train_mask"])[measure]
            min_y = min(y.min(), min_y)
            y_map[value] = list(y)
            l = ax.plot(x, y, label=f"{value}{'*' if is_benchmark else ''}", **kwargs)
            if is_benchmark:
                m = y.mean()
                ax.axhline(m, color=l[-1].get_color(), alpha=0.5, linestyle="--")
        ax.set_ylim([2*min_y-1, 1])
        ax.set_ylabel(measure)
        ax.legend()

        _ax = ax.twinx()
        bar_width, n_bar = 0.1, len(identifiers[element])-1
        X = np.arange(n_fold) - (n_bar-1)/2*bar_width
        max_y = 0
        y_map = pd.DataFrame.from_dict(y_map)
        for col in y_map.columns:
            y_map[col] = y_map[benchmark[element]] - y_map[col]
        y_map.drop(benchmark[element], axis=1, inplace=True)
        for idx, target in enumerate(y_map.columns):
            y = y_map[target]
            max_y = max(max_y, y.max())
            min_y = min(min_y, y.min())
            _ax.bar(X+idx*bar_width, y,
                    label=f"{target}", width=bar_width, linewidth=0, alpha=0.5)
        _ax.set_ylabel("benchmark dominance")
        _ax.set_ylim([min_y, 2*max_y-min_y])
        _ax.axhline(0, color="black", alpha=1, linestyle="--")
        _ax.legend()


# plot
benchmark = best_comb
select_measures = ["accuracy", "precision", "recall", "f1-score"]
n_cols, n_rows = 4, len(select_measures)
fig = plt.figure(figsize=(6*n_cols, 6*n_rows))
fig.suptitle(f"comparison to benchmark {'/'.join(map(str, best_comb.values()))}", fontsize=20)
axes = fig.subplots(n_rows, n_cols)

for idx, measure in enumerate(select_measures):
    compare_plot(axes[idx], benchmark=benchmark, measure=measure)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(output_path)
