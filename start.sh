#!/bin/bash
docker run --rm -it -d --gpus all -p 8888:8888 -v $PWD/:/app/ -v $HOME/clearml.conf:/root/clearml.conf -v $HOME/.clearml:/root/.clearml time_series_playground
