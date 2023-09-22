from pydantic import BaseModel


class TrainConfig(BaseModel):
    input_width: int = 1
    label_width: int = 1
    shift: int = 1
    epochs: int = 20
