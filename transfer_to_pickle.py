import pandas as pd
import math
import pickle

def process_item_attribute_csv(csv_path, pickle_path):
    column_names = ['item', 'attribute1', 'attribute2']
    item_attributes_df = pd.read_csv(csv_path, header=None, names=column_names)

    item_attributes_df["norm"] = item_attributes_df.apply(lambda x: math.sqrt(x["attribute1"]**2 + x["attribute2"]**2), axis=1)

    print(f"number of items: {item_attributes_df['item'].nunique()}")
    print("items information: \nAttribute1:")
    print(item_attributes_df["attribute1"].describe())
    print("Attribute2:")
    print(item_attributes_df["attribute2"].describe())

    item_attributes_dict = item_attributes_df.set_index('item').T.to_dict('list')

    with open(pickle_path, 'wb') as f:
        pickle.dump(item_attributes_dict, f)

csv_path = './datasets/processed_itemAttribute.csv'
pickle_path = './datasets/item_attributes.pickle'
process_item_attribute_csv(csv_path, pickle_path)