import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

traindata = 'competition/MNIST/input/train.csv'
df = pd.read_csv(traindata)

images = df.iloc[:,1:]
labels = df.iloc[:,:1]

# Plot an image

img = images.iloc[1,:].values
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')

# plot a histogram

plt.hist(images.iloc[1,:].values)

# Create lambda function to change value

l = lambda x: 255 if x > 100 else 0

df.apply(l, result_type='broadcast')

# threshold and create new dataframe

threshold = 1
mask = df.my_channel > threshold
column_name = 'my_channel'
df.loc[mask, column_name] = 0


for column in range(0,images.shape[1]):
    for row in range(0,images.shape[0]):
        if images.iloc[row,column]>threshold:
            images.loc[row,column] = 255
        else:
            images.loc[row, column] = 0

