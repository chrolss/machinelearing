import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

traindata = 'competition/MNIST/train.csv'
df = pd.read_csv(traindata)

images = df.iloc[:,1:]
labels = df.iloc[:,:1]

# Plot an image

img = images.iloc[1,:].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')

# plot a histogram

plt.hist(images.iloc[1,:].as_matrix())

# threshold and create new dataframe

threshold = 120

for column in range(0,images.shape[1]):
    for row in range(0,images.shape[0]):
        if images.iloc[row,column]>threshold:
            images.iloc[row,column] = 255
        else:
            images.iloc[row, column] = 0

