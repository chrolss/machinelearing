import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Define function


def showimage(dataframe, imnumber):

    plt.imshow(dataframe.iloc[imnumber].values.reshape(28, 28), cmap='gray')



# Load data

traindata = 'competition/MNIST/input/train.csv'
df = pd.read_csv(traindata)

images = df.iloc[:, 1:]
labels = df.iloc[:, :1]
labels = labels.astype('category')

# Plot an image

img = images.iloc[1, :].values
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray')

# plot a histogram

plt.hist(images.iloc[1, :].values)

# Create lambda function and applymap to change dataframe values

images = images.applymap(lambda x: 255 if x > 150 else 0)

# Create train test split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)
n_rows, n_cols = X_train.shape

# Shape and scale the data
Xnp = X_train.values
Xnp = Xnp / 255
Xnp = Xnp.reshape(-1, n_cols)
Ynp = keras.utils.to_categorical(y_train, num_classes=10)

# Setup the network and corresponding layers
model = Sequential()
model.add(Dense(n_cols, activation='relu', input_dim=784))
model.add(Dropout(0.5))
model.add(Dense(n_cols*2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Define optimizer and compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model to the training data
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(Xnp, Ynp, epochs=5, callbacks=[early_stopping_monitor])

# Evaluate model
Xev = X_test.values
Xev = Xev / 255
Xev = Xev.reshape(-1, n_cols)
Yev = keras.utils.to_categorical(y_test, num_classes=10)
eval = model.evaluate(Xev, Yev, verbose=True)

# Test the predictions

testval = X_test[0, :]
testval = testval.reshape(-1, n_cols)

predictions = model.predict_proba(testval)

def predictimage(image):
    Xp = image.values
    Xp = Xp / 255
    Xp = Xp.reshape(-1, n_cols)
    plt.imshow(Xp.reshape(28, 28),cmap='gray')
    predictions = model.predict_proba(Xp)
    return print(predictions*100)

# Save keras model
model.save_weights("competition/MNIST/model_190317.h5")

# Load model weights
model.load_weights('competition/MNIST/model_190317.h5')
