# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('Testing.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
test = pd.read_csv('Testing.csv')
train = pd.read_csv('Training.csv')
test.sample(5)
train.sample(10)
test.describe()
train.describe()
train.shape
test.shape
!pip install matplotlib seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
train_X = train.iloc[:, :-1]  # First 132 columns as features
train_y = train.iloc[:, -1]   # Last column as target (disease names)
test_X = test.iloc[:, :-1]
test_y = test.iloc[:, -1]
label_encoder = LabelEncoder()
y_encoded_train = label_encoder.fit_transform(train_y)
y_encoded_test = label_encoder.transform(test_y)
model = RandomForestClassifier(n_estimators=1, random_state=0)
model.fit(train_X, y_encoded_train)
pred_y = model.predict(test_X)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_encoded_test, pred_y, average="weighted"):
    print(f"Accuracy: {accuracy_score(y_encoded_test, pred_y):.1f}")
    print(f"Precision: {precision_score(y_encoded_test, pred_y, average=average):.1f}")
    print(f"Recall: {recall_score(y_encoded_test, pred_y, average=average):.1f}")
    print(f"F1-Score: {f1_score(y_encoded_test, pred_y, average=average):.1f}")

evaluate_model(y_encoded_test, pred_y)
joblib.dump(model, "trained_model")
feature_importance = model.feature_importances_
features = np.array(train_X.columns)

sorted_idx = np.argsort(feature_importance)[::-1]
top_n = 10
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx][:top_n], y=features[sorted_idx][:top_n])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top Feature Importance (Random Forest)")
plt.show()