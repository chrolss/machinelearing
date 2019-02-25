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

# Create ECDP (Emperical cumulative distribution function)
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


train_values = 'competition/ML-with-a-heart/input/train_values.csv'
train_labels = 'competition/ML-with-a-heart/input/train_labels.csv'
sample_submission = 'competition/ML-with-a-heart/input/submission_format.csv'

X = pd.read_csv(train_values)
y = pd.read_csv(train_labels)
ss = pd.read_csv(sample_submission)

# Concatenate dataframes to
df = pd.concat([y.heart_disease_present, X], axis=1, join='inner')

# Label Encoding
X.thal = X['thal'].astype('category')
y.heart_disease_present = y.heart_disease_present.astype('category')
X = pd.concat([X, pd.get_dummies(X.thal, 'thal')], axis=1)
X = X.drop(['thal'], axis=1)

# Perform Visual EDA
_ = sns.boxplot(x='heart_disease_present', y='resting_blood_pressure', data=pd.concat([y.heart_disease_present, X], axis=1, join='inner'))
_ = sns.boxplot(x='heart_disease_present', y='serum_cholesterol_mg_per_dl', data=pd.concat([y.heart_disease_present, X], axis=1, join='inner'))

# Perform correlation analysis
sns.set(style='white')
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Scaling the data
patient_id = X.patient_id
X = X.drop(['patient_id'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.drop(['patient_id'], axis=1)

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Setup neural network
n_rows, n_cols = X_train.shape
Xnp = X_train
Xnp = Xnp.reshape(-1,n_cols)
Ynp = pd.get_dummies(y_train).values

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(n_cols,)))
model.add(Dense(30, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(Xnp, Ynp, epochs=30, callbacks=[early_stopping_monitor])

# Evaluate model
Xev = X_test
Xev = Xev.reshape(-1,n_cols)
Yev = pd.get_dummies(y_test).values
eval = model.evaluate(Xev, Yev, verbose=True)

# Make competition predictions

test_values = 'competition/ML-with-a-heart/input/test_values.csv'
testval = pd.read_csv(test_values)
testval.thal = testval['thal'].astype('category')
testval = pd.concat([testval, pd.get_dummies(testval.thal, 'thal')], axis=1)
testval = testval.drop(['thal'], axis=1)

patient_id_test = testval.patient_id
testval = testval.drop(['patient_id'], axis=1)
testval = scaler.fit_transform(testval)
testval = testval.reshape(-1, n_cols)

predictions = model.predict(testval)

# Generate submission

output = pd.DataFrame({'patient_id': patient_id_test,
                       'heart_disease_present': predictions[:, 1]})
output.to_csv('submission_2.csv', index=False)

# Save keras model
model.save_weights("competition/ML-with-a-heart/model_39207.h5")
