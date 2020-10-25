# Project 3

## about

On this project Tobias Sletvold Eilertsen(tei024) and Ingrid Liabakk Eriksen(ier008) worked together. We split the work 50/50.

## Design choices:

Folder structure:

    Input data: /data/

    Python files: /src/

    Main file: /src/main.py

## summary

This project is to create a machine learning model for classifying hand-written digits.

## Technical report

### preprocessing steps

- normalizing

### algorithms

- candidate algorithms were ... based on sklearns documentation.
- the classifiers hyperparameters were either default or set by grid search. We chose the most important or impactful paramterers for the grid search

### performance measure

we measured accuracy in percentage of correct predictions, because a prediction is either right or wrong and the amount of diviation from correct is not important.

### model selection scheme

- we selected models based on the sklearn GridSearchCV class, that both does a grid search over the hyperparameters and does cross validation 

### chosen model

- random forest
- k-neigbors
- linear svc 
- bagging 

### expected performance

### overfitting

- cross validation

### possible improvements
