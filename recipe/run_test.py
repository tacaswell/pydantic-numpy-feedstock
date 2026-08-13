from tempfile import TemporaryDirectory

import numpy as np

from pydantic_numpy.model import NumpyModel
from pydantic_numpy.typing import NpNDArray


class Model(NumpyModel):
    array: NpNDArray
    label: str


with TemporaryDirectory() as directory:
    model = Model(array=np.array([1, 2, 3]), label="test")
    model.dump(directory, "model", pickle=True)
    assert Model.load(directory, "model") == model
