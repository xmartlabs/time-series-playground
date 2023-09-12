# Time Series Playrground

Welcome to Xmartlabs' time series playground. This repository contains scripts and code to train time series models on weather datasets.

## Instructions

* Download the Jena Climate dataset by running:

```bash
./download_jena_dataset.sh
```

* Build the docker container:

```bash
./build.sh
```

* Start the docker container with the Jupyter Notebook:

```bash
./start.sh
```

* Follow the instructions to access the notebook on your browser


## ClearML experiment tracking

If you use the ClearML tracker, make sure to configure correctly your $HOME/clearml.conf file and create a $HOME/.clearml folder that will store the caches and other stuff.
