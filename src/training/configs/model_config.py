from enum import Enum
from pydantic import BaseModel

from models import Baseline, LinearModel, DenseModel, MultiStepDense, ConvModel, RNNModel


class ModelType(str, Enum):
    baseline = 'baseline'
    linear = 'linear'
    dense = 'dense'
    multi_step_dense = 'multi_step_dense'
    convolutional = 'convolutional'
    rnn = 'rnn'

    def model(self, **kwargs):
        if self == ModelType.baseline:
            return Baseline(**kwargs)
        elif self == ModelType.linear:
            return LinearModel(**kwargs)
        elif self == ModelType.dense:
            return DenseModel(**kwargs)
        elif self == ModelType.multi_step_dense:
            return MultiStepDense(**kwargs)
        elif self == ModelType.convolutional:
            return ConvModel(**kwargs)
        elif self == ModelType.rnn:
            return RNNModel(**kwargs)
        else:
            raise ValueError('Invalid model type')


class ModelConfig(BaseModel):
    nn_model_type: ModelType
