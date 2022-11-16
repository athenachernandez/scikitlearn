
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
# matplotlib 3.3.1
from matplotlib import pyplot

import pandas as pd
import numpy as np

def create_df_subset(raw_df, type):
    df_subset = raw_df[raw_df["Type"] == type]
    df_subset = df_subset.sample(frac=1).reset_index(drop=True)

    return df_subset

def main():
    raw_df = pd.read_csv('dataset.csv') # Ask about df naming conventions
    df_subset_0 = create_df_subset(raw_df, 0)
    df_subset_1 = create_df_subset(raw_df, 1)

    # Choose and cut length of dataset to whichever is shorter
    if len(df_subset_0) > len(df_subset_1):
        df_subset_0 = df_subset_0.head(len(df_subset_1))
    else:
        df_subset_1 = df_subset_1.head(len(df_subset_0))

    df = pd.concat([df_subset_0, df_subset_1])
    print(df.head(), len(df))
if __name__ == '__main__':
    main()

# rows_used = int(len(complete_df.index) * .1)

# # df.head()
# print(complete_df)

# rows_used = int(len(complete_df.index) * .1)

# df = complete_df.loc[0:rows_used][:]

# df['Type'] == 1
# # df = df[(df.index > np.percentile(df.index, 10)) & (df.index <= np.percentile(df.index, 10))]
# print(df)

# df.loc[(df['Type'] == 0) & (df['col2'] < value)]
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