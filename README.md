# scikit-learn Project 🦃
- Centered around classifiying whether a website was benign (type 0) or malicious (type 1)
## Features 😎
- Used <b>pandas</b> to create a useable dataset
  - Used an equal about of type 0 and 1 rows
  - Dropped null values
  - Shuffled these values and reset the dataframe's index
- Classifiers with approximately <b>80-90% accuracy</b>
  - Logistic regression
  - Ridge classifier
- Used <b>Matplotlib</b> to graph scatterplots
  - To see relationships and trends between variables I was feeding through SKL's classifiers
  - Red dots represent malicious websites and green dots represent benign websites
## Attempted features 😢
- Because my data isn't really linearly separable, I had trouble with the following classifiers
  - Perceptron
  - Linear Support Vector Machine
- Used Matplotlib to further understand why this was the case
