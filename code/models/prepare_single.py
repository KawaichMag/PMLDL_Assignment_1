from typing import Literal

import torch
from training import Net

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


def preprocess_user_data(
    p_class: int,
    sex: Literal["male", "female"],
    age: int,
    fare: float,
    embarked: Literal["S", "Q", "C"],
    title: Literal["Mr", "Miss"],
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

    return torch.Tensor(data)


data = preprocess_user_data(1, "female", 20, 8, "S", "Mr", "Couple")
model: Net = torch.load("model.pth", weights_only=False)
with torch.no_grad():
    result = model(data)

label = max(result).numpy()
result = result.numpy()
print(f"You will survive with chance of {(label + 1) / 2:.2%}")
