import json
import os
import pandas as pd
from clearml import Dataset


class DatasetLoader():
    """Abstract class that serves to load datasets from different sources (local, ClearML, other tracker)
    """

    def get_dataset_folder(self, dataset_project, dataset_name):
        return NotImplementedError()


class LocalDatasetLoader(DatasetLoader):

    def get_dataset_folder(self, dataset_project, dataset_name):
        return f"data/{dataset_name}"


class ClearMLDatasetLoader(DatasetLoader):

    def get_dataset_folder(self, dataset_project, dataset_name):
        return Dataset.get(dataset_project=dataset_project, dataset_name=dataset_name).get_local_copy()


class JenaDatasetLoader(ClearMLDatasetLoader):
    project = 'Time Series PG'
    dataset = 'jena_climate'

    def load(self):
        self.data_folder = self.get_dataset_folder(self.project, self.dataset)

    def get_data(self):
        assert self.data_folder is not None, "You must call `load` before reading files"
        return pd.read_csv(os.path.join(self.data_folder, "jena_climate_2009_2016.csv"))
