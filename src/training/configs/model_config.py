from enum import Enum
from pydantic import BaseModel, create_model

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

    # Return a dict with all parameters for the model apart from model type
    def model_params(self):
        return self.dict(exclude={'nn_model_type'})

    @classmethod
    def from_config(cls, config):
        model_fields = {k: (type(v), v) for k, v in config['model'].items() if k != 'nn_model_type'}
        ThisModelConfig = create_model('ThisModelConfig', **model_fields, __base__=ModelConfig)
        return ThisModelConfig(**config['model'])
