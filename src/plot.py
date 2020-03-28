#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : plot script for result
# * Last change   : 22:19:02 2020-03-28
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

from pathlib import Path
from itertools import product
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import os

script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
output_dir = script_dir.parent / "doc"
if not os.path.exists(output_dir): os.makedirs(output_dir)
df = pd.read_csv(script_dir.parent / "result" / "trec06p" / "result_03_27_20.csv")
file_format = "pdf"


# get all identifiers
identifiers = ["model", "train_mask", "train_ratio", "use_meta"]
measures = list(set(df.columns) - set(identifiers))
identifiers = dict(zip(identifiers, map(lambda x: np.sort(np.unique(df[x])), identifiers)))
for k,v in identifiers.items(): print(f"{k:12s}{list(v)}")
n_fold = len(identifiers.pop("train_mask"))
identifiers["model"] = identifiers["model"][identifiers["model"] != "naive_bayes_3"] # NOTE: naive_bayes_3 is too deviated from others, don't use it


# select best combination
objective = "f1-score"
combinations = np.array(list(map(lambda x: dict(zip(["model", "train_ratio", "use_meta"], x)),
                product(*identifiers.values()))))
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
        min_y, y_map, color_map = 1.0, {}, {}
        for value in identifiers[element]:
            is_benchmark = value == benchmark[element]
            y = df.loc[np.logical_and.reduce(
                    [df[k] == v for k,v in dict(benchmark, **{element: value}).items()])] \
                    .sort_values(by=["train_mask"])[measure]
            l = ax.plot(x, y, label=f"{value}{'*' if is_benchmark else ''}", **kwargs)
            min_y = min(y.min(), min_y)
            y_map[value] = list(y)
            color_map[value] = l[-1].get_color()
            if is_benchmark:
                m = y.mean()
                ax.axhline(m, color=l[-1].get_color(), alpha=0.5, linestyle="--")
        ax.set_ylim([2.1*min_y-1.1, 1])
        ax.set_ylabel(measure)

        _ax = ax.twinx()
        bar_width, n_bar = 0.2, len(identifiers[element])-1
        X = np.arange(n_fold) - (n_bar-1)/2*bar_width
        max_y = 0
        y_map = pd.DataFrame.from_dict(y_map)
        y_benchmark = y_map[benchmark[element]]
        y_map.drop(benchmark[element], axis=1, inplace=True)
        for col in y_map.columns:
            y_map[col] = y_benchmark - y_map[col]
        for idx, target in enumerate(y_map.columns):
            y = y_map[target]
            max_y = max(max_y, y.max())
            min_y = min(min_y, y.min())
            _ax.bar(X+idx*bar_width, y, color=color_map[target],
                    label=f"diff {target}", width=bar_width, linewidth=0, alpha=0.5)
        _ax.set_ylabel("benchmark difference")
        _ax.set_ylim([min_y*1.1-max_y*0.1, max_y*2.1-min_y*1.1])
        _ax.axhline(0, color="black", alpha=1, linestyle="--", linewidth=0.5)

        ax_handles, ax_labels = ax.get_legend_handles_labels()
        _ax_handles, _ax_labels = _ax.get_legend_handles_labels()
        ax.legend(handles=list(ax_handles)+list(_ax_handles), labels=list(ax_labels)+list(_ax_labels), loc="center right")


# plot comparison
benchmark = best_comb
select_measures = ["accuracy", "precision", "recall", "f1-score"]
n_cols, n_rows = 3, len(select_measures)
fig = plt.figure(figsize=(8*n_cols, 8*n_rows))
fig.suptitle(f"comparison to benchmark {'/'.join(map(str, best_comb.values()))}", fontsize=20)
axes = fig.subplots(n_rows, n_cols)

for idx, measure in enumerate(select_measures):
    compare_plot(axes[idx], benchmark=benchmark, measure=measure)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(output_dir / f"comparison.{file_format}")

# ------------------------------------------------------------------------------------

# best 5 combinations
objective = 'f1-score'
n = 5
best_n_comb = combinations[np.argsort(list(map(lambda x: df.loc[np.logical_and.reduce(
                [df[k] == v for k,v in x.items()])][objective].mean(), combinations)))[::-1][:n]]
print(f"best {n} combinations on {objective}:")
for i in best_n_comb: print("\t", i)


# draw best plot
def best_plot(ax, best_combs, measure="f1-score"):
    x = [f"fold_{i}" for i in range(n_fold)]
    kwargs = {
        "linewidth": 2,
        "markersize": 12,
        "marker": ".",
        "alpha": 0.8,
    }
    for idx, comb in enumerate(best_combs):
        y = df.loc[np.logical_and.reduce(
                [df[k] == v for k,v in comb.items()])] \
                .sort_values(by=["train_mask"])[measure]
        ax.plot(x, y, label=f"best_{idx+1}", **kwargs)
    ax.set_title(f"{measure} on best {len(best_combs)}")
    ax.grid(True)
    ax.set_ylabel(measure)
    ax.legend()


# draw time bar plot
def time_bar(ax, best_combs, n_fold):
    x = list(map(lambda x: f"best_{x}", np.arange(len(best_combs))+1))
    bar_width, n_bar = 0.15, n_fold
    X = np.arange(len(x)) - (n_bar-1)/2*bar_width
    data_train_time = np.array(list(map(lambda comb: df.loc[np.logical_and.reduce(
            [df[k] == v for k,v in comb.items()])] \
            .sort_values(by=["train_mask"])["train_time"], best_combs))).T
    data_test_time = np.array(list(map(lambda comb: df.loc[np.logical_and.reduce(
            [df[k] == v for k,v in comb.items()])] \
            .sort_values(by=["train_mask"])["test_time"], best_combs))).T
    for idx, y in enumerate(data_train_time):
        ax.bar(X+idx*bar_width, y,
                label=f"fold_{idx} train time", width=bar_width, linewidth=0, alpha=0.5)
        ax.bar(X+idx*bar_width, data_test_time[idx], bottom=y,
                label=f"fold_{idx} test time", width=bar_width, linewidth=0, alpha=0.5)

    kwargs = {
        "linestyle": "--",
        "linewidth": 1,
        "markersize": 16,
        "alpha": 0.8,
        "uplims": True,
        "lolims": True,
        "elinewidth": 2,
        "capsize": 6,
    }
    all_time = pd.DataFrame(data_test_time + data_train_time)
    mean_time, min_time, max_time = all_time.mean(), all_time.min(), all_time.max()
    ax.errorbar(X+(n_bar-1)/2*bar_width, mean_time, label=f"total time with error",
                yerr=np.array([mean_time - min_time, max_time - mean_time]), **kwargs)


    ax.set_title(f"time on best {len(x)}")
    ax.grid(True)
    ax.set_ylabel(f"time (s)")
    ax.set_xticks(X+(n_bar-1)/2*bar_width)
    ax.set_xticklabels(x)
    ax.legend()


# plot best
fig.clf()
n_cols, n_rows = 2, 3
fig.set_size_inches(8*n_cols, 8*n_rows)
fig.suptitle(f"meaures on best {n} combinations", fontsize=20)

for idx, measure in enumerate(select_measures):
    best_plot(fig.add_subplot(n_rows, n_cols, idx+1), best_n_comb, measure)
time_bar(fig.add_subplot(n_rows, 1, n_rows), best_n_comb, n_fold)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(output_dir / f"best_{n}.{file_format}")
