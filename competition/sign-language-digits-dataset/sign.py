import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

X_filepath = 'competition/sign-language-digits-dataset/X.npy'
y_filepath = 'competition/sign-language-digits-dataset/Y.npy'


def showpicture(imgarray, row):

    plt.imshow(imgarray[row, :].reshape(100, 100), cmap='gray')


npX = np.load(X_filepath)
npy = np.load(y_filepath)

X = pd.DataFrame(data=npX)
y = pd.DataFrame(data=npy)

plt.hist(X.iloc[0], 10)

l = lambda x: 255 if x > 100 else 0



