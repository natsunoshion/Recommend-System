import os

# 用于训练模型的文本
trainDataset = "../datasets/train.txt"
# 训练集
training_set = "../datasets/train.csv"
# 验证集
valid_set = "../datasets/valid.csv"
# 测试集（无真值）
testDataset = "../datasets/test.txt"
itemAttributeDataset = "../datasets/itemAttribute.txt"
NewItemAttributeDataset = "../datasets/processed_itemAttribute.csv"

DATA_STATISTICS_FOLDER = "../data_statistics/"
# mkdirs
if not os.path.exists(DATA_STATISTICS_FOLDER):
    os.makedirs(DATA_STATISTICS_FOLDER)

PARAMETER_FOLDER = "./middleParameters/"
if not os.path.exists(PARAMETER_FOLDER):
    os.makedirs(PARAMETER_FOLDER)

RESULT_FOLDER = "./results/"
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

item_numFile = DATA_STATISTICS_FOLDER + "item_num.txt"
user_numFile = DATA_STATISTICS_FOLDER + "user_num.txt"
user_dictFile = DATA_STATISTICS_FOLDER + "user_dict.txt"
item_dictFile = DATA_STATISTICS_FOLDER + "item_dict.txt"
item_attrFile = (
    DATA_STATISTICS_FOLDER + "item_attr.txt"
)  # 储存商品属性 {商品实际id: (商品实际id, 属性1, 属性2)......}

# 中间参数
USER_BIAS_VEC = PARAMETER_FOLDER + "UB_VECTOR.dat"
ITEM_BIAS_VEC = PARAMETER_FOLDER + "IB_VECTOR.dat"
P_MATRIX = PARAMETER_FOLDER + "P_MATRIX.dat"
Q_MATRIX = PARAMETER_FOLDER + "Q_MATRIX.dat"
LIL_MATRIX = PARAMETER_FOLDER + "LIL_MATRIX.dat"


# 训练参数
TEST_PORTION = 0.1
FACTORS = 100
EPOCHS = 10
LR = 0.002
LAMBDAUB = 0.01
LAMBDAIB = 0.01
LAMBDAP = 0.007
LAMBDAQ = 0.007
decay_factor = 0.5  # lr衰减
