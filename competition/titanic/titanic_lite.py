import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

testFilepath = "competition/titanic/input/test.csv"
trainFilepath = "competition/titanic/input/train.csv"

df = pd.read_csv(trainFilepath)

def preprocess(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    return df

df = preprocess(df)
df['Age'] = df['Age'].fillna(df['Age'].median())

train_df = df[['PassengerId', 'Sex', 'Age', 'Pclass', 'Survived']]

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['Survived', 'PassengerId'], axis=1),
                                                    train_df['Survived'], test_size=.3)

model = XGBRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

for i in range(len(predictions)):
    predictions[i] = np.round(predictions[i])

score = mean_squared_error(y_test, predictions)

print("Score: ", score)

output = pd.DataFrame({'Predictions': predictions,
                       'True': y_test})

import joblib

joblib.dump(model, 'competition/titanic/titanic_model.pkl')
