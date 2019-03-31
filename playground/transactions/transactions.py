import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

filepath = 'playground/transactions/transactions.csv'

df = pd.read_csv(filepath, delimiter=';')

## Regex Country

dateReg = '/\d{2}-\d{2}-\d{2}'
dateRegSpace = '\s+/\d{2}-\d{2}-\d{2}'
paraReg = '\s+\)+'

df.Specifikation = df.Specifikation.str.replace(dateReg, '')
df.Specifikation = df.Specifikation.str.replace(paraReg, '')
df.Specifikation = df.Specifikation.str.replace(dateRegSpace, '')

## Prepare the columns

df = df.dropna()
df.Item = df.Item.astype('category')

## Prepare the text features

# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit the vector
vec_alphanumeric.fit_transform(df.Specifikation)

### Start the training
# Split out only the text data

df = df.drop(columns='Belopp',axis=0)
X_train, X_test, y_train, y_test = train_test_split(df.Specifikation,
                                                    pd.get_dummies(df.Item),
                                                    random_state=456)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', vec_alphanumeric),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)

# Predict

pl.predict_proba([X_test.iloc[0]])


def predictcategory(spec):
    spec = [spec]
    prob = pl.predict_proba(spec)
    predclass = y_test.columns[prob.argmax()]
    return spec[0] + " is probably " + predclass + " with " + str(prob.max()) + " certainty."

