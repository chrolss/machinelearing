import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.linear_model import LogisticRegression

testFilepath = "competition/titanic/input/test.csv"
trainFilepath = "competition/titanic/input/train.csv"

df = pd.read_csv(trainFilepath)

## Correlation matrix

sns.set(style='white')
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask,cmap=cmap,vmax=.3,center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

## looking for NANs and nulls, find lots in cabin and age
df.isnull().sum()

## drop cabin and ticket and fare (fare and Pclass is highly correlated
df = df.drop(['Cabin'], axis=1)
df = df.drop(['Ticket'], axis=1)
df = df.drop(['Fare'], axis=1)
## Set missing embarked as the most common (Southampton)
df['Embarked'] = df['Embarked'].fillna('S')## working with the data, let's change some
## "Label encode" the sex of the passengers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

## set missing ages as mean age
df['Age'] = df['Age'].fillna(df['Age'].mean())

# One-hot encoding för
Embarked_Encoder = LabelEncoder()
df['Embarked_Encoded'] = Embarked_Encoder.fit_transform(df['Embarked'])
enc = OneHotEncoder(handle_unknown='ignore')
Emb = enc.fit_transform(df['Embarked_Encoded'].values.reshape(-1,1)).toarray()
# put it back into the dataframe
dfOneHot = pd.DataFrame(Emb, columns = ["Emb_"+str(int(i)) for i in range(Emb.shape[1])])
df = pd.concat([df, dfOneHot], axis=1)

## Feature engineering
# 1 - travelling alone

## ISSUE INITIALIING AN EMPTY ARRAY - WHY ?
temp = [0] * len(df)

for i in range(len(temp)):
    if df.SibSp[i] == 0 & df.Parch[i] == 0:
        temp[i] = 1
    else:
        temp[i] = 0

dfTravelAlone = pd.DataFrame(temp, columns = ['Alone'])
df = pd.concat([df, dfTravelAlone], axis=1)


# 2 - Title
Mr = [0] * len(df)
Mrs = [0] * len(df)
Miss = [0] * len(df)

for j in range(len(df)):
    if df.iloc[j,3].__contains__("Mr."):
        Mr[j] = 1
        Mrs[j] = 0
        Miss[j] = 0
    elif df.iloc[j,3].__contains__("Mrs."):
        Mr[j] = 0
        Mrs[j] = 1
        Miss[j] = 0
    elif df.iloc[j,3].__contains__("Miss."):
        Mr[j] = 0
        Mrs[j] = 0
        Miss[j] = 1
    else:
        Mr[j] = 0
        Mrs[j] = 0
        Miss[j] = 0

dfMr = pd.DataFrame(Mr, columns = ['Mr'])
dfMrs = pd.DataFrame(Mrs, columns = ['Mrs'])
dfMiss = pd.DataFrame(Miss, columns = ['Miss'])
df = pd.concat([df, dfMr, dfMrs, dfMiss], axis=1)
# Train Test Split and data prepping

data = df.drop(['Name'],axis=1)
PassId = df['PassengerId']
data = data.drop(['PassengerId'], axis=1)
data = data.drop(['Embarked'], axis=1)
data = data.drop(['Embarked_Encoded'], axis=1)
y = data['Survived']
X = data.drop(['Survived'],axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size = .20)

#%% XGBoost Train model

my_model = xgb.XGBRegressor(n_estimators=20000, learning_rate=0.8)
my_model.fit(train_X, train_y, early_stopping_rounds=300, eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)
for i in range(len(predictions)):
    if predictions[i] > 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

print('Mean Squared Error: ' + str(mean_squared_error(predictions, test_y)))
plt.plot(test_y - predictions)

# Number of incorrect predictions
result = abs(test_y - predictions)
print(1- result.sum()/len(result))

## Logistic Regression

clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=500, multi_class='multinomial').fit(train_X, train_y)
log_predict = clf.predict(test_X)

result = abs(test_y - log_predict)
print(1- result.sum()/len(result))

#%% SVM

svmMod = svm.SVC(gamma='scale')
svmMod.fit(train_X, train_y)

svm_predict = svmMod.predict(test_X)
result = abs(test_y - svm_predict)
print(1- result.sum()/len(result))
#%% Work on test data 2

dft = pd.read_csv(testFilepath)
dft['Sex'] = dft['Sex'].map({'male': 0, 'female': 1})
## drop cabin and ticket and fare (fare and Pclass is highly correlated
dft = dft.drop(['Cabin'], axis=1)
dft = dft.drop(['Ticket'], axis=1)
dft = dft.drop(['Fare'], axis=1)
## set missing ages as mean age
dft['Age'] = dft['Age'].fillna(dft['Age'].mean())
dft['Embarked'] = dft['Embarked'].fillna('S')

# One-hot encoding för
Embarked_Encoder = LabelEncoder()
dft['Embarked_Encoded'] = Embarked_Encoder.fit_transform(dft['Embarked'])
enc = OneHotEncoder(handle_unknown='ignore')
Emb = enc.fit_transform(dft['Embarked_Encoded'].values.reshape(-1,1)).toarray()
# put it back into the dataframe
dfOneHot = pd.DataFrame(Emb, columns = ["Emb_"+str(int(i)) for i in range(Emb.shape[1])])
dft = pd.concat([dft, dfOneHot], axis=1)

## Feature engineering
temp = [0] * len(df)

for i in range(len(temp)):
    if df.SibSp[i] == 0 & df.Parch[i] == 0:
        temp[i] = 1
    else:
        temp[i] = 0

dfTravelAlone = pd.DataFrame(temp, columns = ['Alone'])
df = pd.concat([df, dfTravelAlone], axis=1)


# 2 - Title
Mr = [0] * len(df)
Mrs = [0] * len(df)
Miss = [0] * len(df)

for j in range(len(df)):
    if df.iloc[j,3].__contains__("Mr."):
        Mr[j] = 1
        Mrs[j] = 0
        Miss[j] = 0
    elif df.iloc[j,3].__contains__("Mrs."):
        Mr[j] = 0
        Mrs[j] = 1
        Miss[j] = 0
    elif df.iloc[j,3].__contains__("Miss."):
        Mr[j] = 0
        Mrs[j] = 0
        Miss[j] = 1
    else:
        Mr[j] = 0
        Mrs[j] = 0
        Miss[j] = 0

dfMr = pd.DataFrame(Mr, columns = ['Mr'])
dfMrs = pd.DataFrame(Mrs, columns = ['Mrs'])
dfMiss = pd.DataFrame(Miss, columns = ['Miss'])
dft = pd.concat([df, dfMr, dfMrs, dfMiss], axis=1)

# Train Test Split and data prepping

data = dft.drop(['Name'],axis=1)
PassId = dft['PassengerId']
data = data.drop(['PassengerId'], axis=1)
data = data.drop(['Embarked'], axis=1)
data = data.drop(['Embarked_Encoded'], axis=1)

data = data.reindex(df.columns, axis=1)

predictions = my_model.predict(data.values)
for i in range(len(predictions)):
    if predictions[i] > 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0



output = pd.DataFrame({'PassengerId': dft['PassengerId'],
                       'Survived': predictions})
output.Survived = output.Survived.astype(int)
output.to_csv('submission.csv', index=False)