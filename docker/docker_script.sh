#!/usr/bin/env bash
# **********************************************************************
# * Description   : docker relevant script
# * Last change   : 20:59:43 2020-03-18
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

echo -e "Exp1 script: docker manager..."

WORKING_DIR=`pwd`
SCRIPT_DIR=`dirname "$0"`
SCRIPT_DIR=`cd $SCRIPT_DIR; pwd`
MAIN_DIR=`cd ${SCRIPT_DIR}/../; pwd`

DEFAULT_IMAGE="ml:0.1"
DEFAULT_CONTAINER="exp1"

run_image()
{
    name=$1
    image=$2
    sudo docker run --rm -it \
        --name "$name" \
        --mount type=bind,source=$MAIN_DIR,target=/$name \
        --workdir /$name \
        "$image" \
        bash
}

build_image()
{
    if [ ! -e "${SCRIPT_DIR}/Dockerfile" ] || \
        [ ! -e "${SCRIPT_DIR}/requirements.txt" ] || \
        [ ! -e "${SCRIPT_DIR}/sources.list" ]; then
        echo -e "Abort: file lost"
        exit 1
    fi
    target=$1
    sudo docker build -t "$target" \
        --build-arg USER_ID=$(id -u) \
        --build-arg GROUP_ID=$(id -g) $SCRIPT_DIR
}

load_image()
{
    TAR_PATH=$1
    if [[ ! "$TAR_PATH" = /* ]]; then
        SAVE_PATH=${WORKING_DIR}/${TAR_PATH}
    fi 
    sudo docker load --input "$TAR_PATH"
}

help_info()
{
    echo -e "USAGE: docker_script.sh {TASK} [ARG...]"
}


case "$1" in
    run)
        run_image ${2:-${DEFAULT_CONTAINER}} ${3:-${DEFAULT_IMAGE}}
        ;;
    build)
        build_image ${2:-${DEFAULT_IMAGE}}
        ;;
    load)
        load_image "$2"
        ;;
    *)
        help_info
        ;;
esac
