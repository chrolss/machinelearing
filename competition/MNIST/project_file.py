import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


# Define function


def showimage(dataframe, imnumber):

    plt.imshow(dataframe.iloc[imnumber].values.reshape(28, 28), cmap='gray')



# Load data

traindata = 'competition/MNIST/input/train.csv'
df = pd.read_csv(traindata)

images = df.iloc[:,1:]
labels = df.iloc[:,:1]
labels = labels.astype('category')

# Plot an image

img = images.iloc[1, :].values
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray')

# plot a histogram

plt.hist(images.iloc[1,:].values)

# Create lambda function and applymap to change dataframe values

images = images.applymap(lambda x: 255 if x > 100 else 0)

# Create train test split

X_train, X_test, y_train, y_test = train_test_split(images.values, labels, test_size=0.3)
n_rows, n_cols = X_train.shape

# Shape the data
Xnp = X_train
Xnp = Xnp.reshape(-1, n_cols)
Ynp = pd.get_dummies(y_train)

# Build the neural network

model = Sequential()
model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))
model.add(Dense(n_cols*2, activation='relu'))
model.add(Dense(n_cols, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(Xnp, Ynp, epochs=30, callbacks=[early_stopping_monitor])

# Evaluate model
Xev = X_test
Xev = Xev.reshape(-1,n_cols)
Yev = pd.get_dummies(y_test).values
eval = model.evaluate(Xev, Yev, verbose=True)
