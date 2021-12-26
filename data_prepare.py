# Import Libraries
import os
import pandas as pd

FILE_PATH = os.listdir(r'C:/Users/HP ZBook/Desktop/Python/Datasets/Full/')

dataset = []
label_dict = { "0": 0,"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
               "7": 7, "8": 8}
              # 0 : Aquaculture
              # 1 : Barreland
              # 2 : Croplands
              # 3 : Forests
              # 4 : Grassland
              # 5 : Residential_land
              # 6 : Rice_paddies
              # 7 : Scrub
              # 8 : Water
              
# Read Excel File
for file in FILE_PATH:
    df = pd.read_excel(r'C:/Users/HP ZBook/Desktop/Python/Datasets/Full/' + file, engine='openpyxl')
    df["label"].replace(label_dict, inplace=True)
    dataset.append(df)

data = pd.concat(dataset, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)
idx_train = int(data.shape[0] / 100 * 60)
idx_eval = int(data.shape[0] / 100 * 80)
train_data = data[0:idx_train]
eval_data = data[idx_train + 1:idx_eval]
test_data = data[idx_eval + 1:]

train_data.to_excel('Datasets/Train/Train.xlsx', index=False)
eval_data.to_excel('Datasets/Train/Evaluate.xlsx', index=False)
test_data.to_excel('Datasets/Train/Test.xlsx', index=False)