import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from torch.autograd import Variable


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


def main():
    os.makedirs("data", exist_ok=True)
    with ZipFile("titanic.zip", "r") as zObject:
        zObject.extractall(path="data")

    dataset = pd.read_csv("data/train.csv")
    X_test = pd.read_csv("data/test.csv")

    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Title"] = pd.Series(dataset_title)
    dataset["Title"].value_counts()
    dataset["Title"] = dataset["Title"].replace(
        [
            "Lady",
            "the Countess",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
            "Ms",
            "Mme",
            "Mlle",
        ],
        "Rare",
    )

    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in X_test["Name"]]
    X_test["Title"] = pd.Series(dataset_title)
    X_test["Title"].value_counts()
    X_test["Title"] = X_test["Title"].replace(
        [
            "Lady",
            "the Countess",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
            "Ms",
            "Mme",
            "Mlle",
        ],
        "Rare",
    )

    dataset["FamilyS"] = dataset["SibSp"] + dataset["Parch"] + 1
    X_test["FamilyS"] = X_test["SibSp"] + X_test["Parch"] + 1

    def family(x):
        if x < 2:
            return "Single"
        elif x == 2:
            return "Couple"
        elif x <= 4:
            return "InterM"
        else:
            return "Large"

    dataset["FamilyS"] = dataset["FamilyS"].apply(family)
    X_test["FamilyS"] = X_test["FamilyS"].apply(family)

    dataset["Embarked"].fillna(dataset["Embarked"].mode()[0], inplace=True)
    X_test["Embarked"].fillna(X_test["Embarked"].mode()[0], inplace=True)
    dataset["Age"].fillna(dataset["Age"].median(), inplace=True)
    X_test["Age"].fillna(X_test["Age"].median(), inplace=True)
    X_test["Fare"].fillna(X_test["Fare"].median(), inplace=True)

    dataset = dataset.drop(
        ["PassengerId", "Cabin", "Name", "SibSp", "Parch", "Ticket"], axis=1
    )
    X_test_passengers = X_test["PassengerId"]
    X_test = X_test.drop(
        ["PassengerId", "Cabin", "Name", "SibSp", "Parch", "Ticket"], axis=1
    )

    dataset.to_csv("data/preprocessed_dataset.csv")

    print(dataset.iloc[0:8, :])

    X_train = dataset.iloc[:, 1:9].values
    Y_train = dataset.iloc[:, 0].values
    X_test = X_test.values

    labelencoder_X_1 = LabelEncoder()
    X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
    X_train[:, 4] = labelencoder_X_1.fit_transform(X_train[:, 4])
    X_train[:, 5] = labelencoder_X_1.fit_transform(X_train[:, 5])
    X_train[:, 6] = labelencoder_X_1.fit_transform(X_train[:, 6])

    labelencoder_X_2 = LabelEncoder()
    X_test[:, 1] = labelencoder_X_2.fit_transform(X_test[:, 1])
    X_test[:, 4] = labelencoder_X_2.fit_transform(X_test[:, 4])
    X_test[:, 5] = labelencoder_X_2.fit_transform(X_test[:, 5])
    X_test[:, 6] = labelencoder_X_2.fit_transform(X_test[:, 6])

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), [0, 1, 4, 5, 6]),
        ],
        remainder="passthrough",
    )
    X_train = column_transformer.fit_transform(X_train)
    X_test = column_transformer.transform(X_test)

    print(X_train[:8, :])

    # Ensure dense float arrays (handle sparse outputs)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1)

    net = Net(input_dim=X_train.shape[1])

    batch_size = 50
    num_epochs = 50
    learning_rate = 0.01
    batch_no = len(x_train) // batch_size

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print("Epoch {}".format(epoch + 1))
        x_train, y_train = shuffle(x_train, y_train)
        # Mini batch learning
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            x_var = Variable(torch.FloatTensor(x_train[start:end]))
            y_var = Variable(torch.LongTensor(y_train[start:end]))
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            ypred_var = net(x_var)
            loss = criterion(ypred_var, y_var)
            loss.backward()
            optimizer.step()

    test_var = Variable(torch.FloatTensor(x_val), requires_grad=True)
    with torch.no_grad():
        result = net(test_var)

    values, labels = torch.max(result, 1)
    num_right = np.sum(labels.data.numpy() == y_val)
    print("Accuracy {:.2f}".format(num_right / len(y_val)))

    torch.save(net, "model.pth")


if __name__ == "__main__":
    main()
