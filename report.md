# Project 3

On this project Tobias Sletvold Eilertsen(tei024) and Ingrid Liabakk Eriksen(ier008) worked together. We split the work 50/50.

### Design choices

Folder structure:

    Input data: /data/

    Python files: /src/

    Main file: /src/main.py

Most relevant code is in main.py.

## Summary

The task for this project is to create a machine learning model for classifying hand-written digits. The model use supervised training on a dataset of 70k images of digits to create a model that predicts correct in 98% of cases.
The real life accuracy may be lower depending on how new digits are scanned, lighting, rotation etc, which means the error frequency is minimum 2%. The model will probably have to be improved before it can be used reliably, but machine learning is still the best approach.

## Technical report

### Preprocessing steps
- We normalized the data, because it can speed up the learning of the models and leads to faster convergence. 
- Tried scaling, but had no impact
- difficult to do any feature engineering on pixels

### Algorithms

- candidate algorithms were RandomForestClassifier, Linear support Vector classifier and Multi-layer Perceptron classifier. Models were selected based on sklearns documentation.
- the classifiers hyperparameters were either default or set by grid search. We tried to choose the most important or impactful paramterers for the grid search

### Performance measure

We measured accuracy in percentage of correct predictions, because a prediction is either right or wrong and the amount of diviation from correct is not important.

### Model selection scheme

- we selected models based on the sklearn GridSearchCV class, that both does a grid search over the hyperparameters and does the default 5-fold cross validation to find the best model and hyperparameters.

### Chosen model

Our chosen model is Multi-layer Perceptron classifier, because it gave the highest cross validation score.
MLP is a feed forwardforward (?) neural network with hidden layers containing neurons. MLP Learns features from data and should work better than simpler classifiers as it is dificult to extract features from the data.

### Expected performance

We have a 97,9 percent accuraccy on the test data, and we would expect close to that on unseen data, given that the input pictures are similar in quality, resolution, angle etc. as the pictures we have used for training.

### Overfitting

- Test set is kept away from model until final evaluation

- Cross validation is a preventative measure against overfitting that split the initial training data into smaller training and test sets, and use this sets to tune our models. 

### Possible improvements

- With more processing/time, we could search for more precise hyperparamaters or try out other parameters.
- We could try other classifiers, or bagging classifiers.
