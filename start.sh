#!/bin/bash
docker run --rm -it -u $(id -u):$(id -g) --gpus all -p 8888:8888 -v $PWD/:/app/ time_series_playground
