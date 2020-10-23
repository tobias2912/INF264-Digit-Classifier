import pandas as pd
from sklearn import preprocessing
from sklearn  import  metrics
from sklearn.svm import LinearSVC
from sklearn  import  metrics, preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

def main():
    '''
    loads dataset and learns 3 models. chooses best model and hyperpatrameters
    with grid search and cross validation.
    calculates accuracy of selected model on test set.
    '''
    np.random.seed(123456)
    if False:
        #create a smaller testfile for testing
        digits = get_feature('../data/handwritten_digits_images.csv')
        label = get_label('../data/handwritten_digits_labels.csv')
        create_smaller_file(digits, label)
    else:
        #read given file
        digits=get_feature('../data/digit_smaller.csv')
        label=get_feature('../data/label_smaller.csv')
    
    X_train, Y_train, X_test, Y_test = split(digits, label)

    Y_train = Y_train.reshape(-1,1)
    Y_test = Y_test.reshape(-1,1)
    # train and test different classifiers

    X_train =  preprocessing.normalize(X_train) 
    X_test =  preprocessing.normalize(X_test) 

    # baggingkneighbors
    classifiers = [randomforest(), support_vector() ,kneighbors()]
    best_clf = None
    best_score = 0
    for clf in classifiers:
        clf.fit(X_train, Y_train)
        score = clf.get_grid_search().best_score_
        if score>best_score:
            best_clf = clf
    print('best classifier was ', best_clf)
    #perform prediction with best classifier


def create_smaller_file(X, y):
    '''create and save a smaller file for testing'''
    seed = 33
    X_keep, X_throw, y_keep, y_throw = train_test_split(X, y, 
                                        test_size=0.9, 
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
    '''predicts test valuesprints stats on classifier performance'''
    print("best params: ", clf.best_params_)
    print("predicts test data...")
    predict = clf.predict(X_test)
    score = get_score(predict, Y_test)
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    print('score ',score)
    return score

def randomforest(X_train, Y_train, X_test, Y_test):
    '''Create a randomforest model with cross validation parameter search'''
    tuned_parameters = {"criterion":["gini","entropy"], 'n_estimators':[50, 100, 150]}
    classifier = RandomForestClassifier()
    print("fits classifier...")
    clf = GridSearchCV(classifier, tuned_parameters)
    clf.fit(X_train, Y_train.ravel())
    print(clf.best_score_)   
    return model_stats(clf, X_test, Y_test)

def kneighbors(X_train, Y_train, X_test, Y_test):
    '''Create a K-neighbors model with cross validation parameter search'''
    tuned_parameters = {"n_neighbors":[2, 3,4]}
    classifier = KNeighborsClassifier()
    print("fits classifier...")
    clf = GridSearchCV(classifier, tuned_parameters)
    clf.fit(X_train, Y_train.ravel())
    return model_stats(clf, X_test, Y_test)

def baggingkneighbors(X_train, Y_train, X_test, Y_test):
    '''Create a baggingkneighbors model with cross validation parameter search'''
    tuned_parameters = {'max_samples':[0.5, 1], 'max_features':[0.5, 1]}
    classifier = BaggingClassifier(KNeighborsClassifier(), n_jobs=-1)
    print("fits classifier...")
    clf = GridSearchCV(classifier, tuned_parameters)
    clf.fit(X_train, Y_train.ravel())
    print(clf.best_score_)
    return model_stats(clf, X_test, Y_test)

def support_vector(X_train, Y_train, X_test, Y_test):
    '''Create a K-neighbors model with cross validation parameter search'''
    tuned_parameters = {"penalty":['l1', 'l2']}
    classifier = LinearSVC(max_iter=3000, dual=False)
    print("fits classifier...")
    clf = GridSearchCV(classifier, tuned_parameters)
    clf.fit(X_train, Y_train.ravel())
    return model_stats(clf, X_test, Y_test)

def get_feature(fileName):
    return genfromtxt(fileName, delimiter=',')

def get_label(fileName):
    return genfromtxt(fileName, delimiter='\n')

if __name__ == "__main__":    
    main()