from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from loguru import logger


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 270)
        self.fc2 = nn.Linear(270, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)

        return x


encode_ranges = {
    "p_class": 3,
    "sex": 2,
    "embarked": 3,
    "title": 5,
    "family_size": 4,
}

encode_table = {
    1: 0,
    2: 1,
    3: 2,
    "female": 0,
    "male": 1,
    "S": 2,
    "Q": 1,
    "C": 0,
    "Master": 0,
    "Miss": 1,
    "Mr": 2,
    "Mrs": 3,
    "Couple": 0,
    "Single": 3,
    "InterM": 1,
    "Large": 2,
}


def encode(name, value):
    encoded_list = [0 for i in range(encode_ranges[name])]
    encoded_list[encode_table[value]] = 1
    return encoded_list


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading the model...")
    models["titanic_model"]: Net = torch.load("model.pth", weights_only=False)
    logger.success("Model is loaded")

    yield

    logger.info("Application closes")


app = FastAPI(lifespan=lifespan)


@app.get("/predict_me")
async def predict_me(
    p_class: int,
    sex: Literal["male", "female"],
    age: int,
    fare: float,
    embarked: Literal["S", "Q", "C"],
    title: Literal["Master", "Mr", "Mrs", "Miss"],
    family_size: Literal["Single", "Couple", "InterM", "Large"],
):
    data = (
        encode("p_class", p_class)
        + encode("sex", sex)
        + encode("embarked", embarked)
        + encode("title", title)
        + encode("family_size", family_size)
    )
    data.append(age)
    data.append(fare)

    with torch.no_grad():
        tensor = torch.Tensor(data)
        result = models["titanic_model"](tensor).numpy()
        label = np.where(np.isclose(result, max(result)))[0][0]

    return {"geiven": str(data), "result": str(label)}
