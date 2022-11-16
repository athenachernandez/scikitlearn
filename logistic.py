
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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

    df = pd.concat([df_subset_0, df_subset_1]).dropna()
    df = df.sample(frac=1, random_state=1).reset_index()
    print(df.head(), len(df))

    digitsX = df.iloc[:, [2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    digitsY = df.iloc[:, 21]
    print(digitsX, digitsY)
    trainX, testX, trainY, testY = train_test_split(digitsX, digitsY, test_size = 0.3, shuffle = True) # Do i need to shuffle again

    classifier = LogisticRegression(max_iter = 10000, random_state = 10) # Other paremeters - see if better results, 
    # classifier = RidgeClassifier(max_iter = 10000, random_state = 10) # Other paremeters - see if better results, 
    # SGD Classifier (check out the different loss functions!), Perceptron, Support Vector Machine Classifier (try different kernels!), Linear Support Vector Machine Classifier (simil
    
    classifier.fit(trainX, trainY)
    preds = classifier.predict(testX)

    correct = 0
    incorrect = 0
    for pred, gt in zip(preds, testY):
        if pred == gt: correct += 1
        else: incorrect += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

    plot_confusion_matrix(classifier, testX, testY)
    pyplot.show()

if __name__ == '__main__':
    main()