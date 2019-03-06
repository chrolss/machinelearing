import numpy as np
import os
from PIL import Image

image_filepath = 'competition/sign-language-digits-dataset/images'

X = np.empty((2082, 10000))
y = np.empty((2082, 1))

i = 0

for folder in os.listdir(image_filepath):
    for img in os.listdir(image_filepath + '/' + folder):
        im = Image.open(image_filepath + '/' + folder + '/' + img)
        im = im.resize((100, 100), Image.ANTIALIAS)
        pic = np.array(im)
        pic = np.mean(pic, -1)
        X[i, :] = pic.reshape(1, 10000)
        y[i, 0] = folder
        i += 1

np.save('competition/sign-language-digits-dataset/X.npy', X)
np.save('competition/sign-language-digits-dataset/y.npy', y)



