# **********************************************************************
# * Description   : dockerfile for python3 environment, together with
# *                 docker_script.sh, requirements.txt and source.list
# * Last change   : 21:52:46 2020-03-18
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

# NOTE: python:3 image is under Debian buster
FROM python:3

ARG USER_ID
ARG GROUP_ID

COPY sources.list /etc/apt/sources.list
COPY requirements.txt /requirements.txt

RUN pipSource='https://pypi.tuna.tsinghua.edu.cn/simple' \
    && pip3 config set global.index-url $pipSource \
    && pip3 install -r /requirements.txt -i $pipSource \
    && rm /requirements.txt \
    && addgroup --gid $GROUP_ID user \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# NOTE: uncomment the following code to install with apt and pip3 when needed
# RUN aptDeps='SOME_PACKAGE'; pipDeps='SOME_PACKAGE' \
#     && pipSource='https://pypi.tuna.tsinghua.edu.cn/simple' \
#     && apt update \
#     && apt install -y $aptDeps \
#     && pip3 install $pipDeps -i $pipSource \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt purge -y --auto-remove $aptDeps

USER user
