#!/bin/bash

# constants
DIR=datasets/jena_climate/
DATASET_URL="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

mkdir -p $DIR
wget -P $DIR $DATASET_URL
cd $DIR
unzip jena_climate_2009_2016.csv.zip

# Clean up
rm jena_climate_2009_2016.csv.zip
rm -rf __MACOSX/

echo "Goodbye! Here goes a joke:"
curl -s https://api.chucknorris.io/jokes/random?category=dev | jq -r '.value'
