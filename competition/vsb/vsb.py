import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import seaborn as sns
import pywt as pywt

metadata_test = 'competition/vsb/vsb-power-line-fault-detection/metadata_test.csv'
metadata_train = 'competition/vsb/vsb-power-line-fault-detection/metadata_train.csv'
parquet_train = 'competition/vsb/vsb-power-line-fault-detection/train.parquet'
mdtest = pd.read_csv(metadata_test)
mdtrain = pd.read_csv(metadata_train)

df = pq.read_pandas(parquet_train).to_pandas()
sf = df.head(20)

# These are from a faulty line
plt.subplot(2,1,1)
plt.plot(df['3'])
plt.plot(df['4'])
plt.plot(df['5'])
plt.legend(['Phase 1', 'Phase 2', 'Phase 3'])

# These are from a normal line
plt.subplot(2,1,2)
plt.plot(df['0'])
plt.plot(df['1'])
plt.plot(df['2'])
plt.legend(['Phase 1', 'Phase 2', 'Phase 3'])

# Plot a signal vs. its Fourier transform
ft = fft.fft(df['0'])
plt.plot(ft)
plt.plot(df['0'])
plt.legend(['Fourier Transform', 'Real signal'])