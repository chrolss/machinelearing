import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.

df_raw = pd.read_csv('competition/instant-gratification/input/train.csv')

# Get a list of highly correlated columns with the target column

corr = df_raw.corr()
targetCorr = corr.target.drop(index='target')  # List containing all columns and their correlation to target
targetCorrAbs = targetCorr.apply(lambda x: abs(x))
targetCorrAbsSorted = targetCorrAbs.sort_values(ascending=False)
_ = plt.xticks(rotation=45)
_ = plt.plot(targetCorrAbsSorted.head())
_ = plt.show()

# The most correlated columns are
# 1. wheezy-murtyle-mandrill-entropy
# 2. dorky-turquoise-maltese-important
# 3. muggy-turquoise-donkey-important
# 4. hasty-blue-sheep-contributor
# 5. stinky-olive-kiwi-golden

# Do the competition with limited dataset scope

df = df_raw[targetCorrAbsSorted.head(6).index]
labels = df_raw.target

# Do some scaling for fun
scaler = StandardScaler()
X = scaler.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=.3)

# Start training the LightGBM model

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 20,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval,
                early_stopping_rounds=15)

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval

y_pred_tidy = np.empty_like(y_pred)

for i in range(len(y_pred)):
    if y_pred[i] > 0.499:
        y_pred_tidy[i] = 1
    else:
        y_pred_tidy[i] = 0

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred_tidy) ** 0.5)

# Competition part

df_test_raw = pd.read_csv('../input/test.csv')
df_test = df_test_raw[targetCorrAbsSorted.head().index]

# Do some scaling for fun
scaler = StandardScaler()
X = scaler.fit_transform(df_test)

y_test_pred = gbm.predict(X, num_iteration=gbm.best_iteration)

output = pd.DataFrame({'id': df_test_raw['id'],
                       'target': y_test_pred})

output.target = output.target.apply(lambda x: 1 if x > 0.5 else 0)

output.to_csv('submission.csv', index=False)