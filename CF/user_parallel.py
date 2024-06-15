import os
from utils import *
import math
import time
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor, as_completed


class UserCF:
    def __init__(self, train_p, test_p, model_p):
        self.is_build = False
        self.is_train = False
        self.is_test = False
        self.rated_num = 0  # Total number of ratings
        self.user_matrix = []  # Store user ratings [{itemid: score,...},...]
        self.user_avg = (
            []
        )  # User rating standard(the average score of items)[u1,u2,...]
        self.sim_matrix_user = None  # User similarity matrix (sparse) lil_matrix
        self.item_list = set()  # Item list, using set for deduplication
        self.inverted_item_list = dict()  # Item list reverse index
        self.r = []  # Predicted matrix
        self.train_p = train_p
        self.test_p = test_p
        self.total_sim = 0  # Total similarity, half of the similarity matrix
        self.sim1 = "sim1.user"
        self.sim2 = "sim2.user"
        self.mid = 0
        self.if_2 = False
        self.now_sim = 0
        self.model_p = model_p
        self.topn = 500
        self.thresh = 0
        self.num_threads = 8  # Number of threads in the pool

    def static_analyse(self):
        # use after build
        logging.info(f"user number: {len(self.user_avg)}")
        logging.info(f"item number: {len(self.item_list)}")
        logging.info(f"rated number: {self.rated_num}")
        logging.info(f"total sim number: {self.total_sim}")

    def build(self, path):
        logging.info("Building Rating Matrix...")
        f, user_item = file_read(path)
        temp_count = 0
        score_count = 0
        user_id = None
        user_item_num = None

        for i in user_item:
            if user_id is not None:
                now_item, now_score = int(i[0]), int(i[1])
                self.item_list.add(now_item)
                score_count += now_score
                self.user_matrix[user_id][now_item] = now_score
                temp_count += 1
                if temp_count == user_item_num:
                    self.user_avg[user_id] = score_count / temp_count
                    user_id = None
            else:
                score_count = 0
                user_id, user_item_num = int(i[0]), int(i[1])
                self.rated_num += user_item_num
                while len(self.user_matrix) < user_id + 1:
                    self.user_matrix.append({})
                    self.user_avg.append(0)
                temp_count = 0
        print(len(self.user_avg))

        f.close()
        self.item_list = list(self.item_list)
        self.item_list.sort()
        for x in range(len(self.item_list)):
            self.inverted_item_list[self.item_list[x]] = x
        self.is_build = True
        self.total_sim = int((pow(len(self.user_avg), 2) - len(self.user_avg)) / 2)
        logging.info("Build Rating Matrix Success!")

    def calculate_similarity(self, i, j):
        len_i = len(self.user_matrix[i])
        temp1 = 0
        temp2 = np.sum(
            [
                math.pow(self.user_matrix[i][item] - self.user_avg[i], 2)
                for item in self.user_matrix[i]
            ]
        )
        temp3 = np.sum(
            [
                math.pow(self.user_matrix[j][item] - self.user_avg[j], 2)
                for item in self.user_matrix[j]
            ]
        )
        if len_i <= len(self.user_matrix[j]):
            for item in self.user_matrix[i]:
                if self.user_matrix[j].get(item) is not None:
                    m1 = self.user_matrix[i][item] - self.user_avg[i]
                    m2 = self.user_matrix[j][item] - self.user_avg[j]
                    temp1 += m1 * m2
        else:
            for item in self.user_matrix[j]:
                if self.user_matrix[i].get(item) is not None:
                    m1 = self.user_matrix[i][item] - self.user_avg[i]
                    m2 = self.user_matrix[j][item] - self.user_avg[j]
                    temp1 += m1 * m2

        if temp2 == 0 or temp3 == 0:
            return i, j, 0
        else:
            return i, j, temp1 / (math.sqrt(temp2 * temp3))

    def train(self):
        start = time.time()
        logging.info(f"Start train at {time.asctime(time.localtime(start))}")
        self.sim_matrix_user = [{} for _ in range(len(self.user_avg))]
        self.now_size = 0
        self.mid = int(len(self.user_avg) / 2)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i in range(len(self.user_avg)):
                for j in range(i + 1, len(self.user_avg)):
                    futures.append(executor.submit(self.calculate_similarity, i, j))
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Training Progress"
            ):
                i, j, similarity = future.result()
                self.sim_matrix_user[i][j] = similarity
                self.sim_matrix_user[j][i] = similarity

        for i in range(len(self.user_avg)):
            self.sim_matrix_user[i] = dict(
                sorted(
                    self.sim_matrix_user[i].items(), key=lambda x: x[1], reverse=True
                )
            )

        now_time = time.time()
        logging.info(f"Begin save at {time.asctime(time.localtime(now_time))}")
        model_dir = self.model_p
        save_model(self.sim_matrix_user, os.path.join(model_dir, "sim.user"))
        temp = []
        for i in range(self.mid):
            temp.append(self.sim_matrix_user[i])
            self.sim_matrix_user[i] = {}
        save_model(temp, os.path.join(model_dir, self.sim1))
        temp = []
        for i in range(self.mid, len(self.user_avg)):
            temp.append(self.sim_matrix_user[i])
            self.sim_matrix_user[i] = {}
        save_model(temp, os.path.join(model_dir, self.sim2))
        self.is_train = True
        end = time.time()
        logging.info(
            f"Now is: {time.asctime(time.localtime(end))}, train time cost is {end - start}."
        )

    def predict(self, user, item_j):
        x = 0
        y = 0
        count = 0
        item_j = self.inverted_item_list[item_j]
        if self.if_2:
            user = user - self.mid
        for u in self.sim_matrix_user[user]:
            if (
                self.user_matrix[u].get(item_j) is not None
                and self.sim_matrix_user[user][u] >= self.thresh
            ):
                count += 1
                y += self.sim_matrix_user[user][u]
                x += self.sim_matrix_user[user][u] * (
                    self.user_matrix[u][item_j] - self.user_avg[u]
                )
            if count == self.topn:
                break

        if y == 0:
            return self.user_avg[user]
        else:
            return x / y + self.user_avg[user]

    def test(self, path):
        start = time.time()
        logging.info(f"Start test at {time.asctime(time.localtime(start))}")
        f, data = file_read(path)
        user_id = None
        user_item_num = None
        temp_count = 0
        test_count = 0
        self.if_2 = False
        model_dir = self.model_p
        self.sim_matrix_user = load_model(os.path.join(model_dir, self.sim1))

        for row in tqdm(data, desc="Testing Progress"):
            if user_id is None:
                user_id = int(row[0])
                if user_id >= self.mid and (not self.if_2):
                    del self.sim_matrix_user[:]
                    self.sim_matrix_user = load_model(
                        os.path.join(model_dir, self.sim2)
                    )
                    logging.info("Load over")
                    self.if_2 = True
                while len(self.r) < user_id + 1:
                    self.r.append([])
                user_item_num = 1
                temp_count = 0
                test_count += 1
            else:
                now_item = int(row[1])
                if self.inverted_item_list.get(now_item) is None:
                    p = self.user_avg[user_id]
                else:
                    p = self.predict(user_id, now_item)
                self.r[user_id].append((now_item, p))
                temp_count += 1
                if temp_count == user_item_num:
                    user_id = None
                test_count += 1

        del self.sim_matrix_user[:]
        end = time.time()
        logging.info(
            "%s Test time cost = %fs" % (time.asctime(time.localtime(end)), end - start)
        )

        with open("./results/user_avg_new.txt", "w") as f:
            for i, r in enumerate(self.user_avg):
                f.write(str(i) + " " + str(r) + "\n")

        with open("results/result_cf_user.txt", "w") as f:
            for i in range(len(self.r)):
                f.write(str(i) + "|6\n")
                for j in range(len(self.r[i])):
                    f.write(
                        str(self.r[i][j][0])
                        + " "
                        + str(rate_modify(self.r[i][j][1]))
                        + "\n"
                    )

        logging.info("Successfully writed results!")
        self.is_test = True
        true_ratings = []
        for row in data:
            true_ratings.append(float(row[2]))
        predicted_ratings = []
        for user_ratings in self.r:
            for item, rating in user_ratings:
                predicted_ratings.append(rating)
        rmse = self.calculate_rmse(predicted_ratings, true_ratings)
        logging.info(f"RMSE: {rmse}")

        f.close()

    def calculate_rmse(self, predicted_ratings, true_ratings):
        return math.sqrt(mean_squared_error(predicted_ratings, true_ratings))
