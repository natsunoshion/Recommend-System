import math
import psutil
import os


trainDataset="./datasets/train.txt"
testDataset="./datasets/test.txt" # test set (no ground truth)
itemAttributeDataset="./datasets/itemAttribute.txt"
NewItemAttributeDataset="./datasets/processed_itemAttribute.csv"

DATA_STATISTICS_FOLDER= './data_statistics/'
if not os.path.exists(DATA_STATISTICS_FOLDER):
    os.makedirs(DATA_STATISTICS_FOLDER)

item_numFile=DATA_STATISTICS_FOLDER+'item_num.txt'
user_numFile=DATA_STATISTICS_FOLDER+'user_num.txt'
user_dictFile = DATA_STATISTICS_FOLDER+'user_dict.txt'
item_dictFile =  DATA_STATISTICS_FOLDER+'item_dict.txt'
item_attrFile =  DATA_STATISTICS_FOLDER+'item_attr.txt' # 储存商品属性 {商品实际id: (商品实际id, 属性1, 属性2)......}

def getProcessMemory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss
    return memory_usage/1024/1024 # MB

# analyse the whole dataset
def analyseWhloeDataset():
    print("正在统计itemAttribute.txt信息... ")
    with open(itemAttributeDataset, 'r') as f:
        item_attr_data= f.readlines()
        f.close()

    item_num_train=0
    item_num_test=0
    user_num_train=0
    user_num_test=0
    item_num = 0
    user_num = 0
    user_dict = {}
    item_dict = {}
    item_attr = {}
    whloe_attr_data = []
    data_with_attribute_count=0

    min_user_id = 0
    max_user_id = 0
    min_item_id = 0
    max_item_id = 0

    # if attribute is None, then replace it with -1
    for attr_data in item_attr_data:
        line = attr_data.strip()
        if line == "":
            continue
        item_id, attr1, attr2 = line.split('|')
        if attr1 == 'None':
            attr1 = -1
        else:
            attr1 = int(attr1)
        if attr2 == 'None':
            attr2 = -1
        else:
            attr2 = int(attr2)
        item_id_int = int(item_id)
        max_item_id =max(item_id_int, max_item_id)
        min_item_id = min(item_id_int, min_item_id)
        # 建立商品的实际id和程序中id的映射
        item_dict[item_id_int] = item_num
        # 储存商品属性 {商品实际id: (商品实际id, 属性1, 属性2)......}
        item_attr[item_id_int] = (item_id_int, attr1, attr2)
        # whloe_attr_data = raw itemattribute data
        whloe_attr_data.append((item_id_int, attr1, attr2))
        item_num += 1
        data_with_attribute_count += 1

    #TODO use -1 to replace the missing value(unseen item)?
    attr_list = []
    item_no_attrs = 0
    for i in range(min_item_id, max_item_id + 1):
        try:
            attr_list.append([item_attr[i][1], item_attr[i][2]])
        except KeyError:
            # use the mean value to replace the missing value
            attr_list.append([-1, -1])
            item_no_attrs += 1

    print("正在统计train.txt信息...")
    #统计用户id的最大最小值 建立数据集用户id（可能不连续，同PR）和在程序中id的映射
    with open(trainDataset, 'r') as f:
        wholeDataset = f.readlines()
        f.close()

    user_ratings = {}  # 用于储存用户评分：{ 用户1: [评分1, 评分2, ...], ... }
    item_ratings = {}  # 用于储存商品评分: { 商品1: [评分1, 评分2, ...], ... }
    train_data_num = 0 #记录train.txt数据条数
    Rate = 0.0
    rating_scale=[0, 100]
    # 记录该分数的打分人数
    num_of_this_rating = [0 for _ in range(0, rating_scale[1] - rating_scale[0] + 1)]
    whole_set_mean_rating = 0.0



    for data in wholeDataset:
        line = data.strip()
        if line.find('|') != -1:
            user_id, user_item_count = line.split('|')
            user_id_int = int(user_id)
            try:
                u = user_dict[user_id_int]
            except KeyError:
                user_ratings[user_id_int] = []
                user_dict[user_id_int] = user_num
                max_user_id=max(user_id_int, max_user_id)
                min_user_id=min(user_id_int, min_user_id)
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
                i =item_dict[item_id_int]
            except KeyError:
                item_dict[item_id_int] = item_num
                max_item_id=max(item_id_int, max_item_id)
                min_item_id =min(item_id_int, min_item_id)
                item_num += 1
                item_num_train += 1
            try:
                item_ratings[item_id_int].append(Rate)
            except KeyError:
                item_ratings[item_id_int] = []
                item_ratings[item_id_int].append(Rate)

    whole_set_mean_rating /= train_data_num
    print("train处理完毕")

    print("正在统计test.txt信息...")
    with open(testDataset, 'r') as f:
        test_data = f.readlines()
        f.close()

    test_data_num = 0
    for test in test_data:
        line = test.strip()
        if line.find('|') != -1:
            user_id, user_item_count = line.split('|')
            user_id_int = int(user_id)
            try:
                u =user_dict[user_id_int]
            except KeyError:
                user_ratings[user_id_int] = []
                user_dict[user_id_int] =user_num
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
                i =item_dict[item_id_int]
            except KeyError:
                item_dict[item_id_int] =item_num
                max_item_id = max(item_id_int, max_item_id)
                min_item_id = min(item_id_int, min_item_id)
                item_num += 1
                item_num_test += 1


    print("test处理完毕")

     #储存“物品实际 id，attr1，attr2”的形式（id从：0到max_itemid连续）
    print("正在写入到processed_itemAttribute.csv...")
    with open(NewItemAttributeDataset, 'w') as f:
        for i in range(len(attr_list)):
            f.write('%d,%d,%d\n' % (i, attr_list[i][0], attr_list[i][1]))
        f.close()
    print("写入完毕")

    print("正在生成统计信息文件...")
    with open(DATA_STATISTICS_FOLDER+'result.txt', 'w',encoding='utf-8') as statf:
        statf.write('存在的用户总数: %d\n' % user_num)
        statf.write('存在的商品总数: %d\n' % item_num)
        statf.write('train.txt评分总数: %d\n' % train_data_num)
        statf.write('train.txt所有得分的平均值: %f\n' %whole_set_mean_rating)
        statf.write('属性集数据条数: %d\n' % data_with_attribute_count)
        statf.close()
    print("已写出result.txt")

    with open(DATA_STATISTICS_FOLDER+'userAvgScore.csv', 'w') as f:
        s = sorted(user_ratings.items(), key=lambda x: x[0])
        for u, r in s:
            f.write(str(u))#usrid(注意不一定所有人都打分了）
            f.write(',')
            total_sum = 0.0
            for rate in r:
                total_sum += rate
            if len(r) != 0:
                mean = total_sum / len(r)
            else:
                mean = 0.0
            f.write(str(mean)) #该用户打出的平均分
            f.write(',')
            f.write(str(len(r))) #该用户给多少个商品进行了打分
            f.write('\n')
        f.close()
    print("已写出userAvgScore.csv; 格式：usrid:avgScore:ratingTimes")

    item_avg_list=[]#用于画出商品信息
    item_being_rated_times=[]#商品被打分次数
    with open(DATA_STATISTICS_FOLDER+'itemAgScore.csv', 'w') as f:
        s = sorted(item_ratings.items(), key=lambda x: x[0])
        for i, r in s:
            f.write(str(i)) #itemID
            f.write(',')
            total_sum = 0.0
            for rate in r:
                total_sum += rate
            if len(r) != 0:
                mean = total_sum / len(r)
            else:
                mean = 0.0
            format(float(mean), '.2f')
            item_avg_list.append(mean)
            f.write(str(mean)) #该商品获得的平均分
            f.write(',')
            f.write(str(len(r))) #有多少个用户给这个商品打分
            item_being_rated_times.append(len(r))
            f.write('\n')
        f.close()



    print("已写出itemAgScore.csv; 格式：item_id:avgScore:ratedTimes")
    #写出打分分布情况
    with open(DATA_STATISTICS_FOLDER+'ratingDistribution.csv', 'w') as rate:
        for i, j in enumerate(num_of_this_rating):
            rate.write('%d,%d\n' % (i + rating_scale[0], j))
        rate.close()
    print("已写出ratingDistribution.csv")
    print("正在写出usr_dict, item_dict, item_attr, item_num, usr_num信息")
    with open(user_dictFile,'w') as f:
        for key, value in user_dict.items():
            f.write(f"{key}: {value}\n")
        f.close()

    with open(item_dictFile,'w') as f:
        for key,value in item_dict.items():
            f.write(f"{key}: {value}\n")
        f.close()

    with open(item_attrFile,'w') as f:
        for key,value in item_attr.items():
            f.write(f"{key},{value[0]},{value[1]},{value[2]}\n")
        f.close()
    with open(user_numFile,'w') as f:
        f.write(str(user_num))
        f.close()
    with open(item_numFile,'w') as f:
        f.write(str(item_num))
        f.close()

    print('写出完毕')

    import matplotlib.pyplot as plt

    # Plotting item average score distribution
    hist, bins, _ = plt.hist(item_avg_list, bins=10, rwidth=0.9)
    plt.xticks(bins)  # Set x-axis ticks to bin edges
    plt.xlabel('Average Score')
    plt.ylabel('Number of Items')
    plt.title('Distribution of Item Average Scores')
    plt.savefig(DATA_STATISTICS_FOLDER+'item_avg_score_distribution.pdf')

analyseWhloeDataset()
memory_usage = getProcessMemory()
print(f"Analyse data memory usage: {memory_usage} MB")

