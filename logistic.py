
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
# matplotlib 3.3.1
from matplotlib import pyplot

import pandas as pd
import numpy as np

complete_df = pd.read_csv('dataset.csv')
# df.head()
print(complete_df)

rows_used = int(len(complete_df.index) * .1)

df = complete_df.loc[0:rows_used][:]

df['Type'] == 1
# df = df[(df.index > np.percentile(df.index, 10)) & (df.index <= np.percentile(df.index, 10))]
print(df)

df.loc[(df['Type'] == 0) & (df['col2'] < value)]


# create 0 df, grab 10%; create 1 df, grab 10%; when creating test df training will shuffle randomize


## Leave out first 10% or 90% for both benign and malicious

# digits = load_digits()
# digitsX = digits.images.reshape(len(digits.images), 64)
# digitsY = digits.target
# trainX, testX, trainY, testY = train_test_split(
#     digitsX, digitsY, test_size = 0.3, shuffle = True
#     )

# classifier = LogisticRegression(max_iter = 10000)
# classifier.fit(trainX, trainY)
# preds = classifier.predict(testX)

# correct = 0
# incorrect = 0
# for pred, gt in zip(preds, testY):
#     if pred == gt: correct += 1
#     else: incorrect += 1
# print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

# plot_confusion_matrix(classifier, testX, testY)
# pyplot.show()