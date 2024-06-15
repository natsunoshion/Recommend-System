import math
import psutil
import os

trainDataset = "./datasets/train.txt"
testDataset = "./datasets/test.txt"  # test set (no ground truth)
itemAttributeDataset = "./datasets/itemAttribute.txt"
NewItemAttributeDataset = "./datasets/processed_itemAttribute.csv"

DATA_STATISTICS_FOLDER = "./data_statistics/"
if not os.path.exists(DATA_STATISTICS_FOLDER):
    os.makedirs(DATA_STATISTICS_FOLDER)

item_numFile = DATA_STATISTICS_FOLDER + "item_num.txt"
user_numFile = DATA_STATISTICS_FOLDER + "user_num.txt"
user_dictFile = DATA_STATISTICS_FOLDER + "user_dict.txt"
item_dictFile = DATA_STATISTICS_FOLDER + "item_dict.txt"
item_attrFile = (
    DATA_STATISTICS_FOLDER + "item_attr.txt"
)  # Store item attributes {actual item id: (actual item id, attribute1, attribute2)......}


def get_process_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss
    return memory_usage / 1024 / 1024  # MB


# Analyse the whole dataset
def analyse_whloe_dataset():
    print("Analyzing itemAttribute.txt information...")
    with open(itemAttributeDataset, "r") as f:
        item_attr_data = f.readlines()
        f.close()

    item_num_train = 0
    item_num_test = 0
    user_num_train = 0
    user_num_test = 0
    item_num = 0
    user_num = 0
    user_dict = {}
    item_dict = {}
    item_attr = {}
    whloe_attr_data = []
    data_with_attribute_count = 0

    min_user_id = 0
    max_user_id = 0
    min_item_id = 0
    max_item_id = 0

    # if attribute is None, then replace it with -1
    for attr_data in item_attr_data:
        line = attr_data.strip()
        if line == "":
            continue
        item_id, attr1, attr2 = line.split("|")
        if attr1 == "None":
            attr1 = -1
        else:
            attr1 = int(attr1)
        if attr2 == "None":
            attr2 = -1
        else:
            attr2 = int(attr2)
        item_id_int = int(item_id)
        max_item_id = max(item_id_int, max_item_id)
        min_item_id = min(item_id_int, min_item_id)
        # Map actual item id to program id
        item_dict[item_id_int] = item_num
        # Store item attributes {actual item id: (actual item id, attribute1, attribute2)......}
        item_attr[item_id_int] = (item_id_int, attr1, attr2)
        # whloe_attr_data = raw itemattribute data
        whloe_attr_data.append((item_id_int, attr1, attr2))
        item_num += 1
        data_with_attribute_count += 1

    # TODO use -1 to replace the missing value(unseen item)?
    attr_list = []
    item_no_attrs = 0
    for i in range(min_item_id, max_item_id + 1):
        try:
            attr_list.append([item_attr[i][1], item_attr[i][2]])
        except KeyError:
            # use the mean value to replace the missing value
            attr_list.append([-1, -1])
            item_no_attrs += 1

    print("Analyzing train.txt information...")
    # Collect the max and min user id, map the dataset user id (possibly non-continuous, same as PR) to the program id
    with open(trainDataset, "r") as f:
        wholeDataset = f.readlines()
        f.close()

    user_ratings = {}  # Store user ratings: { user1: [rating1, rating2, ...], ... }
    item_ratings = {}  # Store item ratings: { item1: [rating1, rating2, ...], ... }
    train_data_num = 0  # Record the number of train.txt data entries
    Rate = 0.0
    rating_scale = [0, 100]
    # Record the number of ratings for each score
    num_of_this_rating = [0 for _ in range(0, rating_scale[1] - rating_scale[0] + 1)]
    whole_set_mean_rating = 0.0

    for data in wholeDataset:
        line = data.strip()
        if line.find("|") != -1:
            user_id, user_item_count = line.split("|")
            user_id_int = int(user_id)
            try:
                u = user_dict[user_id_int]
            except KeyError:
                user_ratings[user_id_int] = []
                user_dict[user_id_int] = user_num
                max_user_id = max(user_id_int, max_user_id)
                min_user_id = min(user_id_int, min_user_id)
                user_num += 1
                user_num_train += 1
        else:
            if line == "":
                continue
            item_id, rating_str = line.split()
            train_data_num += 1
            item_id_int = int(item_id)
            Rate = float(rating_str)
            num_of_this_rating[math.floor(Rate)] += 1
            whole_set_mean_rating += Rate
            user_ratings[user_id_int].append(Rate)
            try:
                i = item_dict[item_id_int]
            except KeyError:
                item_dict[item_id_int] = item_num
                max_item_id = max(item_id_int, max_item_id)
                min_item_id = min(item_id_int, min_item_id)
                item_num += 1
                item_num_train += 1
            try:
                item_ratings[item_id_int].append(Rate)
            except KeyError:
                item_ratings[item_id_int] = []
                item_ratings[item_id_int].append(Rate)

    whole_set_mean_rating /= train_data_num
    print("Finished processing train")

    print("Analyzing test.txt information...")
    with open(testDataset, "r") as f:
        test_data = f.readlines()
        f.close()

    test_data_num = 0
    for test in test_data:
        line = test.strip()
        if line.find("|") != -1:
            user_id, user_item_count = line.split("|")
            user_id_int = int(user_id)
            try:
                u = user_dict[user_id_int]
            except KeyError:
                user_ratings[user_id_int] = []
                user_dict[user_id_int] = user_num
                max_user_id = max(user_id_int, max_user_id)
                min_user_id = min(user_id_int, min_user_id)
                user_num += 1
                user_num_test += 1
        else:
            if line == "":
                continue
            test_data_num += 1
            item_id_int = int(line)
            try:
                i = item_dict[item_id_int]
            except KeyError:
                item_dict[item_id_int] = item_num
                max_item_id = max(item_id_int, max_item_id)
                min_item_id = min(item_id_int, min_item_id)
                item_num += 1
                item_num_test += 1

    print("Finished processing test")

    # Store "item actual id, attr1, attr2" in the form (id from: 0 to max_itemid continuous)
    print("Writing to processed_itemAttribute.csv...")
    with open(NewItemAttributeDataset, "w") as f:
        for i in range(len(attr_list)):
            f.write("%d,%d,%d\n" % (i, attr_list[i][0], attr_list[i][1]))
        f.close()
    print("Writing complete")

    print("Generating statistics files...")
    with open(DATA_STATISTICS_FOLDER + "result.txt", "w", encoding="utf-8") as statf:
        statf.write("Total number of users: %d\n" % user_num)
        statf.write("Total number of items: %d\n" % item_num)
        statf.write("Total number of ratings in train.txt: %d\n" % train_data_num)
        statf.write(
            "Average score of all ratings in train.txt: %f\n" % whole_set_mean_rating
        )
        statf.write(
            "Number of attribute data entries: %d\n" % data_with_attribute_count
        )
        statf.close()
    print("result.txt written")

    with open(DATA_STATISTICS_FOLDER + "userAvgScore.csv", "w") as f:
        s = sorted(user_ratings.items(), key=lambda x: x[0])
        for u, r in s:
            f.write(str(u))  # userid (note that not all users have rated)
            f.write(",")
            total_sum = 0.0
            for rate in r:
                total_sum += rate
            if len(r) != 0:
                mean = total_sum / len(r)
            else:
                mean = 0.0
            f.write(str(mean))  # Average score given by the user
            f.write(",")
            f.write(str(len(r)))  # Number of ratings given by the user
            f.write("\n")
        f.close()
    print("userAvgScore.csv written; Format: userid:avgScore:ratingTimes")

    item_avg_list = []  # For plotting item information
    item_being_rated_times = []  # Number of times the item has been rated
    with open(DATA_STATISTICS_FOLDER + "itemAgScore.csv", "w") as f:
        s = sorted(item_ratings.items(), key=lambda x: x[0])
        for i, r in s:
            f.write(str(i))  # itemID
            f.write(",")
            total_sum += rate
            if len(r) != 0:
                mean = total_sum / len(r)
            else:
                mean = 0.0
            format(float(mean), ".2f")
            item_avg_list.append(mean)
            f.write(str(mean))  # Average score received by the item
            f.write(",")
            f.write(str(len(r)))  # Number of users who rated the item
            item_being_rated_times.append(len(r))
            f.write("\n")
        f.close()

    print("itemAgScore.csv written; Format: item_id:avgScore:ratedTimes")
    # Write out rating distribution
    with open(DATA_STATISTICS_FOLDER + "ratingDistribution.csv", "w") as rate:
        for i, j in enumerate(num_of_this_rating):
            rate.write("%d,%d\n" % (i + rating_scale[0], j))
        rate.close()
    print("ratingDistribution.csv written")
    print("Writing user_dict, item_dict, item_attr, item_num, user_num information")
    with open(user_dictFile, "w") as f:
        for key, value in user_dict.items():
            f.write(f"{key}: {value}\n")
        f.close()

    with open(item_dictFile, "w") as f:
        for key, value in item_dict.items():
            f.write(f"{key}: {value}\n")
        f.close()

    with open(item_attrFile, "w") as f:
        for key, value in item_attr.items():
            f.write(f"{key},{value[0]},{value[1]},{value[2]}\n")
        f.close()
    with open(user_numFile, "w") as f:
        f.write(str(user_num))
        f.close()
    with open(item_numFile, "w") as f:
        f.write(str(item_num))
        f.close()

    print("Writing complete")

    import matplotlib.pyplot as plt

    # Plotting item average score distribution
    hist, bins, _ = plt.hist(item_avg_list, bins=10, rwidth=0.9)
    plt.xticks(bins)  # Set x-axis ticks to bin edges
    plt.xlabel("Average Score")
    plt.ylabel("Number of Items")
    plt.title("Distribution of Item Average Scores")
    plt.savefig(DATA_STATISTICS_FOLDER + "item_avg_score_distribution.pdf")


analyse_whloe_dataset()
memory_usage = get_process_memory()
print(f"Analyse data memory usage: {memory_usage} MB")
