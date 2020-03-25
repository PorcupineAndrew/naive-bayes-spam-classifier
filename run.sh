#!/usr/bin/env bash
# **********************************************************************
# * Description   : run experiment script for Exp1
# * Last change   : 22:58:53 2020-03-25
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

echo -e "Exp1 script: run experiment..."

WORKING_DIR=`pwd`
SCRIPT_DIR=`dirname "$0"`
SCRIPT_DIR=`cd $SCRIPT_DIR; pwd`
MAIN_DIR=`cd ${SCRIPT_DIR}/src; pwd`
DATA_DIR=`cd ${SCRIPT_DIR}/data; pwd`
RESULT_DIR=${SCRIPT_DIR}/result

[ ! -d "$RESULT_DIR" ] && mkdir "$RESULT_DIR"
RESULT_DIR=`cd ${RESULT_DIR}; pwd`
cd $MAIN_DIR

run_data_process()
{
    DATASET=$1
    NUM_WORKER=$2
    shift; shift;
    FLAGS=$@

    ./data_process.py \
        --dataset "$DATASET" \
        --num-workers "$NUM_WORKER" $FLAGS
}

run_classifier()
{
    DATASET=$1
    NUM_WORKER=$2
    TRAIN_RATIO=$3
    MODEL=$4
    USE_META=$5
    shift; shift; shift; shift; shift;
    FLAGS=$@

    [ ! -d "${RESULT_DIR}/${DATASET}" ] && mkdir "${RESULT_DIR}/${DATASET}"
    ./run_classifier.py \
        --dataset "$DATASET" \
        --n-workers "$NUM_WORKER" \
        --train-ratio "$TRAIN_RATIO" \
        --output-dir "${RESULT_DIR}/${DATASET}" \
        --model "$MODEL" \
        --use-meta "$USE_META" $FLAGS
}

help_info()
{
    echo -e "USAGE: run.sh {TASK} [ARG...]"
}

TASK=$1
DATASET=${2:-trec06p}
NUM_WORKER=${3:-20}
shift; shift; shift;

case "$TASK" in
    run)
        run_classifier "$DATASET" "$NUM_WORKER" $@
        ;;
    all)
        run_classifier "$DATASET" "$NUM_WORKER" "_" "_" "_" "--all-test"
        ;;
    preprocess)
        run_data_process "$DATASET" "$NUM_WORKER" $@
        ;;
    *)
        help_info
        ;;
esac
