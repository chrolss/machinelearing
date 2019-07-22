import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
import json

# Load the model

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/model_weights_220719.h5")
print("Loaded model from disk")

# Load the test values

with open('input/test_patient.json', 'r') as file:
    data = json.load(file)

testval = pd.DataFrame(data, index=[0])

# When reading a larger test file, we get at least one of each thal type, but now
# with only one patient, we have to manually fill in the dummy labels
thal_value = testval.thal.iloc[0]
testval['thal_reversible_defect'] = 0
testval['thal_fixed_defect'] = 0
testval['thal_normal'] = 0

if thal_value == 'reversible_defect':
    testval['thal_reversible_defect'] = 1
elif thal_value == 'normal':
    testval['thal_normal'] = 1
elif thal_value == 'fixed_defect':
    testval['thal_fixed_defect'] = 1
else:
    print("no valid thal value")

testval = testval.drop(['thal'], axis=1)
patient_id_test = testval.patient_id
testval = testval.drop(['patient_id'], axis=1)

# Scale the test values
train_values = 'input/train_values.csv'
X = pd.read_csv(train_values)

# Label Encoding
X.thal = X['thal'].astype('category')
X = pd.concat([X, pd.get_dummies(X.thal, 'thal')], axis=1)
X = X.drop(['thal'], axis=1)

# Scaling the data
X = X.drop(['patient_id'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

patient_id = testval.patient_id
testval = testval.drop(['patient_id'], axis=1)
testval = scaler.transform(testval)
testval = testval.reshape(-1, 15)

predictions = model.predict(testval)
