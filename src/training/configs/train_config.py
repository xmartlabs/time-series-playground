from pydantic import BaseModel


class TrainConfig(BaseModel):
    input_width: int = 1
    label_width: int = 1
    shift: int = 1
    epochs: int = 20
    learning_rate: float = 1e-3
    sequence_stride: int = 96
