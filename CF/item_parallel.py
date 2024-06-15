import time
import numpy as np
import pandas as pd
import pickle
from utils import *
import math
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os


def calculate_rmse(predicted_ratings, true_ratings):
    return math.sqrt(mean_squared_error(predicted_ratings, true_ratings))

class CollaborativeFiltering:
    def __init__(self, train_path, test_path, attribute_path, model_directory):
        self.similarity_map = {}
        self.attribute_similarity = {}
        self.train_path = train_path
        self.test_path = test_path
        self.attribute_path = attribute_path
        self.split_size = 0.1
        self.train_data = {}
        self.test_data = {}
        self.user_item_train_data = {}
        self.item_user_train_data = {}
        self.item_attributes = []
        self.train_data = []
        self.valid_data = []
        self.bias = {}
        self.num_of_users = 0
        self.num_of_items = 0
        self.num_of_ratings = 0
        self.topn = 500
        self.model_directory = model_directory
        self.is_build = False
        self.is_train = False
        self.is_test = False

    def fetch_similarity(self, item_i, item_j):
        if item_i in self.similarity_map and item_j in self.similarity_map[item_i]:
            return self.similarity_map[item_i][item_j]
        elif item_j in self.similarity_map and item_i in self.similarity_map[item_j]:
            return self.similarity_map[item_j][item_i]
        return None

    def calculate_similarity(self, user, item_i):
        bias_i = self.bias["deviation_of_item"][item_i] + self.bias["overall_mean"]
        similarity_dict = {}
        for item_j in self.user_item_train_data[user].keys():
            similarity_dict[item_j] = self.fetch_similarity(item_i, item_j)
            if similarity_dict[item_j] is None:
                bias_j = self.bias["deviation_of_item"].get(item_j, 0) + self.bias["overall_mean"]

                if self.item_attributes[item_i][2] == -1 or self.item_attributes[item_j][2] == -1:
                    attribute_similarity = 0
                else:
                    attribute_similarity = (self.item_attributes[item_i][0] * self.item_attributes[item_j][0]
                                            + self.item_attributes[item_i][1] * self.item_attributes[item_j][1]) \
                                            / (self.item_attributes[item_i][2] * self.item_attributes[item_j][2])
                norm_i = 0
                norm_j = np.sum([math.pow(self.item_user_train_data[item_j][user] - bias_j, 2) for user in self.item_user_train_data[item_j].keys()])
                sim_ = 0
                count = 0
                if item_i in self.item_user_train_data:
                    for same_user, score in self.item_user_train_data[item_i].items():
                        norm_i += math.pow(self.item_user_train_data[item_i][same_user] - bias_i, 2)
                        if same_user not in self.item_user_train_data[item_j]:
                            continue
                        count += 1
                        sim_ += (self.item_user_train_data[item_i][same_user] - bias_i) \
                            * (self.item_user_train_data[item_j][same_user] - bias_j)
                    if count < 20:
                        sim_ = 0
                    if sim_ != 0:
                        sim_ /= math.sqrt(norm_i * norm_j)
                similarity = (sim_ + attribute_similarity) / 2
                if item_i not in self.similarity_map:
                    self.similarity_map[item_i] = {}
                self.similarity_map[item_i][item_j] = similarity
                similarity_dict[item_j] = similarity
        return similarity_dict

    def collaborative_filtering_bias(self):
        predicted_ratings = []
        futures = []
        num_threads = 4
        print(f"Using {num_threads} threads for parallel processing.")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for index, row in enumerate(self.valid_data.values):
                futures.append(executor.submit(self.process_row, row, index, predicted_ratings))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Collaborative Filtering Bias"):
                future.result()
                if len(predicted_ratings) % 500 == 0 and len(predicted_ratings) != 0:
                    logging.info(f"Predicted {len(predicted_ratings)} ratings")
                    logging.info(f"RMSE: {calculate_rmse(predicted_ratings, self.valid_data['score'][:len(predicted_ratings)])}")
        logging.info(f"RMSE: {calculate_rmse(predicted_ratings, self.valid_data['score'])}")

    def process_row(self, row, index, predicted_ratings):
        user, item_i, true_rating = row
        rating = 0
        similar_items = self.calculate_similarity(user, item_i)
        similar_items = sorted(similar_items.items(), key=lambda x: x[1], reverse=True)
        bias_i = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_i] + self.bias['deviation_of_user'][user]
        norm = 0
        for i, (item_j, similarity) in enumerate(similar_items):
            if i > self.topn:
                break
            bias_j = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_j] + self.bias['deviation_of_user'][user]
            rating += similarity * (self.item_user_train_data[item_j][user] - bias_j)
            norm += similarity
        if norm != 0:
            rating /= norm
        rating += bias_i
        predicted_ratings.append(valid_rate(rating))
        if index == len(self.valid_data.values) - 1:
            file_save(self.similarity_map, os.path.join(self.model_directory, "similarity_map.pickle"))

    def predict(self):
        index = 0
        predictions = defaultdict(dict)
        futures = []
        num_threads = os.cpu_count()  # Set the number of threads to the number of CPU cores
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for user, items in self.test_data.items():
                futures.append(executor.submit(self.process_user, user, items, predictions))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Prediction"):
                future.result()
        with open(os.path.join(self.model_directory, 'result_CF_bias.txt'), 'w') as f:
            for user, items in predictions.items():
                f.write(f"{user}|6\n")
                for item, score in items.items():
                    f.write(f"{item} {score}\n")

    def process_user(self, user, items, predictions):
        for item_i in items:
            rating = 0
            if item_i in self.bias["deviation_of_item"]:
                similar_items = self.calculate_similarity(user, item_i)
                similar_items = sorted(similar_items.items(), key=lambda x: x[1], reverse=True)
                bias_i = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_i] + self.bias['deviation_of_user'][user]
                norm = 0
                for i, (item_j, similarity) in enumerate(similar_items):
                    if i > self.topn:
                        break
                    bias_j = self.bias['overall_mean'] + self.bias['deviation_of_item'].get(item_j, 0) + self.bias['deviation_of_user'][user]
                    rating += similarity * (self.item_user_train_data[item_j][user] - bias_j)
                    norm += similarity
                if norm != 0:
                    rating /= norm
                rating += bias_i
            else:
                rating = self.bias['overall_mean'] + self.bias['deviation_of_user'][user]
            predictions[user][item_i] = rate_modify(rating)

    def exec(self):
        self.valid_data = pd.read_csv('../datasets/valid.csv')
        self.train_data = pd.read_csv('../datasets/train.csv')
        with open("../datasets/item_user_train.pickle", 'rb') as f:
            self.item_user_train_data = pickle.load(f)
        with open("../datasets/user_item_train.pickle", 'rb') as f:
            self.user_item_train_data = pickle.load(f)
        with open("../datasets/item_attributes.pickle", 'rb') as f:
            self.item_attributes = pickle.load(f)
        with open("../datasets/bias.pickle", 'rb') as f:
            self.bias = pickle.load(f)
        with open("../datasets/test_data.pickle", 'rb') as f:
            self.test_data = pickle.load(f)

        # train collaborative filtering model
        print('Start collaborative filtering model')
        start = time.time()
        self.collaborative_filtering_bias()
        end = time.time()
        logging.info(f'Running time: {end - start} Seconds')

        # logging.info('Start prediction')
        # start = time.time()
        # self.predict()
        # end = time.time()
        # logging.info(f'Running time: {end - start} Seconds')

if __name__ == '__main__':
    base_dir = './experiments/itemcf'

    # is_retrain = input("Do you want to retrain the model? (y/n): ").strip().lower()
    # assert is_retrain in ['y', 'n'], "Invalid input! Please type 'y' or 'n'."
    # if is_retrain == 'y':
    #     archive_existing_directory(base_dir)

    log_directory = os.path.join(base_dir, 'logs')
    model_directory = os.path.join(base_dir, 'models')

    setup_logging(log_directory, 'training')

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model = CollaborativeFiltering('../datasets/train.txt', '../datasets/test.txt', '../datasets/itemAttribute.txt', model_directory)
    model.exec()