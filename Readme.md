# Naive Bayes Classifier

#### 作者： 陈熠豪

#### 学号：2017011486

#### 邮箱： chenyiha17@mails.tsinghua.edu.cn

## 如何复现

-   建立运行环境

    -   下面提供一个基于 docker 建立运行环境的方法。你也可以使用任何其他方式构建运行环境，注意本仓库需要 Python3.7 及以上，所需的 package 请见 `./docker/requirements.txt`

    -   如果使用 docker，请注意替换 docker 源为国内源，否则可能导致镜像拉取失败

    -   运行下面的命令，构建一个 docker image。这个脚本命令会拉取 Python3 镜像，替换 Apt 和 pip3 源，安装本仓库所需的所有 package，并建立必要的用户和用户组

        ```bash
        ./docker/docker_script.sh build
        ```

    -   运行下面的命令，建立运行时 container。这个脚本命令会使用刚刚构建的镜像来建立容器，并运行 bash。得到的容器会挂载本仓库的文件目录，可以直接交互，容器在退出后会自动删除

        ```bash
        ./docker/docker_script.sh run
        ```

-   放置数据文件

    -   请将 trec06p 数据目录放置在 `./data` 下

        ```
        data
        └── trec06p
            ├── data
            └── label
        ```

-   运行实验

    -   运行下面的命令，得到数据预处理结果

        ```
        ./run.sh preprocess trec06p [NUM_WORKER]
        ```

        其中 NUM_WORKER 是一个可选参数，表示多进程的最大并行数，默认为 20。请根据运行机器的实际情况选择合适的数值

        数据预处理的结果将将存放在 `./data/trec06p/preprocess` 目录下，大小约为 158MB

    -   运行下面的命令，得到**所有**实验结果。

        ```bash
        ./run.sh all trec06p [NUM_WORKER]
        ```

        其中 NUM_WORKER 是一个可选参数，表示多进程的最大数量，默认为 20。请根据运行机器的实际情况选择合适的数值

        这条命令将运行 30 个组合的对比实验，在 5 折交叉测试下总计为 150 个 task，耗时较长，建议 NUM_WORKER 在可能的范围内尽可能大

        得到的结果将存于 `./result/trec06p/result_{TIME}.csv`

        参考：机器逻辑内核数为 22，NUM_WORKER=20 时，运行总耗时约为 30min，内存占用峰值约为 6GB

    -   如果觉得运行所有实验太麻烦，只是为了验证代码的可行性，可以运行以下命令进行检查

        ```bash
        ./run.sh check
        ```

        它将运行 (naive_bayes_1, 1.0, true) 的单个实验，进行 5 折交叉验证并得到结果，同时有比较好看的输出

## 仓库文件组成

```
.
├── data
│   └── trec06p
├── doc
│   ├── best_5.pdf
│   ├── best_5.png
│   ├── comparison.pdf
│   ├── comparison.png
│   └── report.md
├── docker
│   ├── Dockerfile
│   ├── docker_script.sh
│   ├── requirements.txt
│   └── sources.list
├── Readme.md
├── report.pdf
├── result
│   └── trec06p
├── run.sh
└── src
    ├── classifier.py
    ├── data_process.py
    └── plot.py
```

-   `./data/` 存放数据文件目录

-   `./doc/` 存放分析图表、report 源文件

-   `./docker/` 存放 docker 构建相关文件和脚本

-   `./result/` 存放运行结果

-   `./src/` 存放源代码
