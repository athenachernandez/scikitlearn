
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
    df = df.sample(frac=1, random_state=1).reset_index()
    print(df.head(), len(df))

    # Split data into test and training
    # Perform what type of classification
    # Graph ? Check Dr. J's GitHub

if __name__ == '__main__':
    main()