import numpy as np
import pandas as pd

data_path = r'C:/Users/HP ZBook/Desktop/Python/Datasets/Full/Acquaculture_7_Bands.xlsx'
img_path = r'C:/Users/HP ZBook/Desktop/Python/GEE-DEMO/LANDSAT8_4_7_band_new_Projec21.tif'


df = pd.read_excel(data_path)
# print(df)

X_labels = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7']
y_label = ['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[X_labels], df[y_label], train_size=0.8 ,test_size=0.2, stratify=df['Name'], random_state=0)

# # ###
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold

# clf=RandomForestClassifier(n_estimators=100, random_state=9)

# skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=12)

# cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')

# print('Accuracy score - 10foldCV - Training dataset: {}'.format(np.mean(cv_scores)))

###
# from sklearn.ensemble import RandomForestClassifier
# clf=RandomForestClassifier(n_estimators=10, random_state=9)
# clf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)

# from sklearn.metrics import accuracy_score, confusion_matrix
# print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))

# ##
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import StratifiedKFold, GridSearchCV

# clf=RandomForestClassifier(random_state=9)
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
# param_grid = { 
#     'n_estimators': [10, 100],
#     'max_features': ['auto', 'sqrt'],
#     'criterion' : ['gini','entropy'],
# }

# gridCV_clf = GridSearchCV(estimator = clf, param_grid=param_grid, cv = skf, scoring='accuracy', verbose=2)
# gridCV_clf.fit(X_train, y_train)
# print(gridCV_clf.best_params_)
# print(gridCV_clf.best_score_)

##
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features='auto', n_estimators=100, random_state=9, criterion='entropy')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))
