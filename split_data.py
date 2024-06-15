import numpy as np
import pandas as pd
import math
from CF.utils import file_read, file_save

item_user_train_data = {}
train_data = {}
user_item_train_data = {}
split_size = 0.1
attribute_path = "./datasets/itemAttribute.txt"
train_train_data = []
train_test_data = []
bias = {}
num_of_user = 0
num_of_item = 0
num_of_rate = 0


def static_analyse(u, i, r):
    """Prints the statistics of users, items, and ratings."""
    print(f"user number: {u}")
    print(f"item number: {i}")
    print(f"rated number: {r}")


def load_train_data():
    global num_of_rate, num_of_user, num_of_item, train_data
    user_item = file_read("./datasets/train.txt")
    for line in user_item:
        line = line.strip()
        if "|" in line:
            user, rates_num = map(int, line.split("|")[:2])
            num_of_rate += rates_num
            num_of_user += 1
        else:
            item, rate = map(int, line.split())
            if item not in train_data:
                train_data[item] = {}
                num_of_item += 1
            train_data[item][user] = rate
    static_analyse(num_of_user, num_of_item, num_of_rate)
    file_save(train_data, "./datasets/train_data.pickle")


def train_test_split():
    global train_train_data, train_test_data, item_user_train_data, user_item_train_data
    for item, rates in train_data.items():
        for index, (user, score) in enumerate(rates.items()):
            if index == 0:
                train_train_data.append([user, item, score])
                item_user_train_data.setdefault(item, {})[user] = score
                user_item_train_data.setdefault(user, {})[item] = score
            elif np.random.rand() < split_size:
                train_test_data.append([user, item, score])
            else:
                train_train_data.append([user, item, score])
                item_user_train_data.setdefault(item, {})[user] = score
                user_item_train_data.setdefault(user, {})[item] = score

    train_test_data_df = pd.DataFrame(
        train_test_data, columns=["user", "item", "score"]
    )
    train_test_data_df.to_csv("./datasets/valid.csv", index=False)
    train_train_data_df = pd.DataFrame(
        train_train_data, columns=["user", "item", "score"]
    )
    train_train_data_df.to_csv("./datasets/train.csv", index=False)
    file_save(item_user_train_data, "./datasets/item_user_train.pickle")
    file_save(user_item_train_data, "./datasets/user_item_train.pickle")

    print("train test data")
    static_analyse(
        train_test_data_df["user"].nunique(),
        train_test_data_df["item"].nunique(),
        len(train_test_data_df),
    )
    print("train train data")
    static_analyse(
        train_train_data_df["user"].nunique(),
        train_train_data_df["item"].nunique(),
        len(train_train_data_df),
    )


def process_item_attribute():
    attr = file_read(attribute_path)
    item_attributes = []
    for line in attr:
        line = line.strip()
        item, attr1, attr2 = line.split("|")
        item = int(item)
        attr1 = None if attr1 == "None" else int(attr1)
        attr2 = None if attr2 == "None" else int(attr2)
        item_attributes.append([item, attr1, attr2])

    item_attributes_df = pd.DataFrame(
        item_attributes, columns=["item", "attribute1", "attribute2"]
    )
    item_attributes_df["attribute1"].fillna(0, inplace=True)
    item_attributes_df["attribute2"].fillna(0, inplace=True)
    item_attributes_df["norm"] = item_attributes_df.apply(
        lambda x: math.sqrt(x["attribute1"] ** 2 + x["attribute2"] ** 2), axis=1
    )

    print(f"number of items: {item_attributes_df['item'].nunique()}")
    print("items information: \nAttribute1:")
    print(item_attributes_df["attribute1"].describe())
    print("Attribute2:")
    print(item_attributes_df["attribute2"].describe())

    item_attributes_dict = item_attributes_df.set_index("item").T.to_dict("list")
    file_save(item_attributes_dict, "./datasets/item_attributes.pickle")


def calculate_data_bias():
    train_train_data_df = pd.DataFrame(
        train_train_data, columns=["user", "item", "score"]
    )
    overall_mean = train_train_data_df["score"].mean()
    deviation_of_user = (
        train_train_data_df.groupby("user")["score"].mean() - overall_mean
    )
    deviation_of_item = (
        train_train_data_df.groupby("item")["score"].mean() - overall_mean
    )
    bias["overall_mean"] = overall_mean
    bias["deviation_of_user"] = deviation_of_user.to_dict()
    bias["deviation_of_item"] = deviation_of_item.to_dict()
    file_save(bias, "./datasets/bias.pickle")


def load_test_data():
    test_data = {}
    test_lines = file_read("./datasets/test.txt")
    num_of_user = 0
    num_of_rate = 0
    items = []
    for line in test_lines:
        line = line.strip()
        if "|" in line:
            user, rates = map(int, line.split("|")[:2])
            num_of_rate += rates
            num_of_user += 1
            test_data[user] = []
        else:
            item = int(line)
            items.append(item)
            test_data[user].append(item)
    static_analyse(num_of_user, len(set(items)), num_of_rate)
    file_save(test_data, "./datasets/test_data.pickle")


def print_section_header(message):
    print("\n" + "=" * 50)
    print(f"= {message}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print_section_header("Start loading train data")
    load_train_data()
    print_section_header("Start dividing train data")
    train_test_split()
    print_section_header("Load and process item attribute")
    process_item_attribute()
    print_section_header("Start calculating data bias")
    calculate_data_bias()
    print_section_header("Start loading test data")
    load_test_data()
