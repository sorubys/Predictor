# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import LabelEncoder
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Import dataset
data = 'asiacup.csv'
df = pd.read_csv(data)
# asiacup_dataset = pd.read_csv('asiacup.csv')

# # Encoding Categorical Columns
# df.replace({'Team': {'Pakistan': 0, 'Sri Lanka': 1, 'India': 2, 'Bangladesh': 3, 'Hong Kong': 4, 'UAE': 5, 'Afghanistan': 6},
#                      'Opponent': {'Pakistan': 0, 'Sri Lanka': 1, 'India': 2, 'Bangladesh': 3, 'Hong Kong': 4, 'UAE': 5, 'Afghanistan': 6},
#                      'Ground': {'Sharjah': 0, 'Colombo(PSS)': 1, 'Moratuwa': 2, 'Kandy': 3, 'Colombo(SSC)': 4, 'Dhaka': 5, 'Chattogram': 6, 'Chandigarh': 7, 'Cuttack': 8, 'Kolkata': 9, 
#                      'Colombo(RPS)': 10, 'Dambulla': 11, 'Lahore': 12, 'Karachi': 13, 'Mirpur': 14, 'Fatullah': 15, 'Dubai(DSC)': 16, 'Abu Dhabi': 17, },
#                      'Toss': {'Lose': 0, 'Win': 1},
#                      'Selection': {'Batting': 0, 'Bowling': 1}},
#                      inplace=True)

label_encoder = LabelEncoder()
df['Team'] = label_encoder.fit_transform(df['Team'])
print("Mapping of labels to integers:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {index}")

df['Opponent'] = label_encoder.fit_transform(df['Opponent'])
print("Mapping of labels to integers:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {index}")

df['Ground'] = label_encoder.fit_transform(df['Ground'])
print("Mapping of labels to integers:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {index}")

df['Toss'] = label_encoder.fit_transform(df['Toss'])
print("Mapping of labels to integers:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {index}")

df['Selection'] = label_encoder.fit_transform(df['Selection'])
print("Mapping of labels to integers:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {index}")

df.info()
df.head()
df.describe()
df.isnull().sum()
#extracting the feature and target arrays
X = df.drop(['Result'], axis=1)
y = df['Result']
label_encoder = LabelEncoder()
y_cleaned = [label.strip() for label in y]  # Clean labels
y_encoded = label_encoder.fit_transform(y_cleaned)  # Convert to integers

# X.head()
# y_encoded.head()

import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=X,label=y_encoded)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.3, random_state = 1)

# import XGBClassifier
from xgboost import XGBClassifier

# declare parameters
params = {
            'objective':'binary:logistic', 
            'max_depth': 5, 
            'colsample_bytree': 0.3,
            'alpha': 10, 
            'learning_rate': 0.1, 
            'n_estimators':100 
        }

# instantiate the classifier
xgb_clf = XGBClassifier(**params)

# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)



# # make predictions on test data
y_pred = xgb_clf.predict(X_test)

# # check accuracy score
from sklearn.metrics import accuracy_score

from xgboost import cv
from sklearn.metrics import roc_auc_score

params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
               'max_depth': 5, 'alpha': 10}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)

# # Find the best number of boosting rounds
best_num_boost_rounds = xgb_cv['test-auc-mean'].idxmax()
print(f'Best number of boosting rounds: {best_num_boost_rounds}')

final_model = xgb.train(
   params=params,
   dtrain=data_dmatrix,
   num_boost_round=best_num_boost_rounds
 )


# # Convert the test set into DMatrix
dtest = xgb.DMatrix(data=X_test)

# # Predict the test set
y_pred_prob = final_model.predict(dtest)

auc_score = roc_auc_score(y_test, y_pred_prob)
print(f'AUC Score: {auc_score:.4f}')

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(xgb_clf, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


