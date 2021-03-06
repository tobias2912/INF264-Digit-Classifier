import pandas as pd
from sklearn  import  metrics, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from numpy import genfromtxt
from classifiers import *
import matplotlib.pyplot as plt
import time

def main():
    '''
    loads dataset and learns 3 models. chooses best model and hyperpatrameters
    with grid search and cross validation.
    calculates accuracy of selected model on test set.
    '''
    np.random.seed(123456)
    digits, label = get_data(False)
    X_train, Y_train, X_test, Y_test = split(digits, label)
    
    Y_train = Y_train.reshape(-1,1)
    Y_test = Y_test.reshape(-1,1)

    X_train =  preprocessing.normalize(X_train) 
    X_test =  preprocessing.normalize(X_test) 
    
    #X_train = preprocessing.scale(X_train)
    #X_test = preprocessing.scale(X_test)

    # train and test different classifiers
    classifiers = [MLP_classifier(),randomforest(), support_vector()] 
    best_clf = None
    best_score = 0
    start = time.time()
    for clf in classifiers:
        print('fits classifier...', type(clf).__name__)
        clf.fit(X_train, Y_train)
        score = clf.get_grid_search().best_score_
        print('score:', score)
        if score>best_score:
            best_clf = clf
        print(f'best params{clf.get_grid_search().best_params_}')
        totaltime = time.time() - start
        print('fitting finished in ', totaltime, 'seconds')
        model_stats(clf.get_grid_search(), X_test, Y_test)
    print('best classifier was ', best_clf)

    #perform prediction with best classifier
    start = time.time()
    pred_test = best_clf.get_grid_search().predict(X_test)
    pred_score = get_score(pred_test, Y_test)
    totaltime = time.time() - start
    print('predicted test score: ', pred_score)
    print('prediction took ', totaltime, 'seconds')
    
    #plot the confusion matrix for the best classifier
    #model_stats(best_clf.get_grid_search(), X_test, Y_test)

def get_data(full_dataset):
    '''read files and return either full dataset or a smaller for testing.
    can also create smaller files'''
    if full_dataset:
        digits = get_feature('../data/handwritten_digits_images.csv')
        label = get_label('../data/handwritten_digits_labels.csv')
        #create_smaller_file(digits, label)
    else:
        #read given file
        digits = get_feature('../data/digit_smaller.csv')
        label = get_feature('../data/label_smaller.csv')
    return digits, label

def create_smaller_file(X, y):
    '''create and save a smaller file for testing'''
    seed = 33
    X_keep, X_throw, y_keep, y_throw = train_test_split(X, y, 
                                        test_size=0.80, 
                                        shuffle=True, 
                                        random_state=seed)
    np.savetxt('../data/digit_smaller.csv',X_keep , delimiter=',', fmt='%f')
    np.savetxt('../data/label_smaller.csv',y_keep , delimiter='\n', fmt='%f')
 

def split(digits, label):
    '''split dataset into test, train, val'''
    seed = 33
    X_train, X_test, Y_train, Y_test = train_test_split(digits, label,      
                                                                test_size=0.3, 
                                                                shuffle=True, 
                                                                random_state=seed)
    return X_train, Y_train, X_test, Y_test
       

def get_score(predict, actual):
    '''calcuate percentage of correct predictions'''
    correct, wrong = 0,0
    for x, y in enumerate(actual):
        if y == predict[x]:
            correct+=1
        else:
            wrong+=1
    return correct/(correct+wrong)

def plot(digits, label, predict):
    '''Plot a single image and print the label to console'''
    img = digits.reshape(digits.shape[0], 28, 28)
    plt.imshow(img[predict], cmap="Greys")
    print(label[predict])
    plt.show()
      

def model_stats(clf, X_test, Y_test):
    '''plots confusion matrix'''
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test, normalize='true')
    disp.figure_.suptitle(f"Confusion Matrix for classifier:{clf}")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.show()

def get_feature(fileName):
    return genfromtxt(fileName, delimiter=',')

def get_label(fileName):
    return genfromtxt(fileName, delimiter='\n')

if __name__ == "__main__":    
    main()