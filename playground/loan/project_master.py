import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

filepath = 'loans.csv'
df = pd.read_csv(filepath)

df['annual_income_natural_log'] = df['annual_income_natural_log'].fillna((df['annual_income_natural_log']).mean())
df['num_days_with_credit_line'] = df['num_days_with_credit_line']\
    .fillna((df['num_days_with_credit_line']).mean())
df['num_inquiries_by_creditors_6mths'] = df['num_inquiries_by_creditors_6mths'].\
    fillna(0)

df['loan_purpose'] = df['loan_purpose'].map({'all_other': 0, 'home_improvement': 1, 'small_business': 2, 'credit_card': 3, 'debt_consolidation': 4, 'major_purchase': 5, 'educational': 6})

X = df.drop(['not_paid'], axis=1)
y = df['not_paid']

train_X, train_y, test_X, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=.2)

