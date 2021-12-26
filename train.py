# Import Libraries
import numpy as np
import pandas as pd
import xgboost as xgb

# Read Excel File
train_data = pd.read_excel('Datasets/Train/Train.xlsx', engine='openpyxl')
eval_data = pd.read_excel('Datasets/Train/Evaluate.xlsx', engine='openpyxl')
test_data = pd.read_excel('Datasets/Train/Test.xlsx', engine='openpyxl')

# Prepare Datasets
train_label = train_data['label'].values
train_data = train_data.loc[:, ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5',
                                'Band_6', 'Band_7']]
eval_label = eval_data['label'].values
eval_data = eval_data.loc[:, ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5',
                              'Band_6', 'Band_7']]
test_label = test_data['label'].values
test_data = test_data.loc[:, ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5',
                              'Band_6', 'Band_7']]


##### Train with XGBoost XGBRegressor
# from xgboost import XGBClassifier
# from sklearn.metrics import mean_absolute_error
# Setup model
# my_model_3 = XGBClassifier(n_estimators =20, learning_rate = 0.05, n_jobs = 4)
# Train model
# my_model_3.fit(train_data,train_label,
#              early_stopping_rounds = 5,
#              eval_set= [(eval_data,eval_label)],
#             verbose = False)
# Get predictions
# predictions_3 = my_model_3.predict(eval_data)
# mae_3 = mean_absolute_error(eval_label, predictions_3)
# print(mae_3)
#####


##### Train with RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# clf=RandomForestClassifier(n_estimators=100, random_state=9)
# skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=12)
# clf.fit(train_data,train_label)
# y_pred=clf.predict(eval_label)
# from sklearn.metrics import accuracy_score, confusion_matrix
# print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))
#####

#### Train with xgboost 
dtrain = xgb.DMatrix(train_data, label=train_label, missing=np.nan)
deval = xgb.DMatrix(eval_data, label=eval_label, missing=np.nan)

# Declare Parameters
param = {'max_depth': 2, 'eta': 0.3, 'objective': 'multi:softmax', 'nthread': 4,
         'num_class': 10, 'shuffle': True}
evallist = [(deval, 'eval'), (dtrain, 'train')]
num_round = 100

# Training
bst = xgb.train(param, dtrain, num_round, evallist)

# Testing
dtest = xgb.DMatrix(test_data)
labelpred = bst.predict(dtest)
print('=' * 100)
print("Prediction Accuracy: " + str(int(np.sum(labelpred == test_label) / labelpred.shape[0] * 100)) + " %")

# Save Model
bst.save_model('model/XGBoost.model')
