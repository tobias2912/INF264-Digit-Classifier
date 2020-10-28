import pandas as pd
from sklearn  import  metrics, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from numpy import genfromtxt
from classifiers import *
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

    #X_train_norm =  preprocessing.normalize(X_train) 
    #X_test_norm =  preprocessing.normalize(X_test) 
    
    #X_train_scaled = preprocessing.scale(X_train)
    #X_test_scaled = preprocessing.scale(X_test)

    # train and test different classifiers
    classifiers = [randomforest(), support_vector(), MLP_classifier()] 
    best_clf = None
    best_score = 0
    start = time.time()
    for clf in classifiers:
        print('fits classifier...', type(clf).__name__)
        #clf.fit(X_train_scaled, Y_train)   vet ikke om man trenger å fitte alle modellene??
        #clf.fit(X_train_norm, Y_train)
        clf.fit(X_train, Y_train)
        score = clf.get_grid_search().best_score_
        print('score:', score)
        if score>best_score:
            best_clf = clf
        print(f'best params{clf.get_grid_search().best_params_}')
        totaltime = time.time() - start
        print('fitting finished in ', totaltime, 'seconds')
    print('best classifier was ', best_clf)

    #perform prediction with best classifier
    start = time.time()
    pred_test = best_clf.get_grid_search().predict(X_test)
    #pred_test_scaled = best_clf.get_grid_search().predict(X_test_scaled)
    #pred_test_norm = best_clf.get_grid_search().predict(X_test_norm)
    pred_score = get_score(pred_test, Y_test)
    #pred_score_scaled = get_score(pred_test_scaled, Y_test)
    #pred_score_norm = get_score(pred_test_norm, Y_test)
    totaltime = time.time() - start
    print('predicted test score: ', pred_score)
    #print('predicted test score on scaled data: ', pred_score_scaled)
    #print('predicted test score on normalized data: ', pred_score_norm)
    print('prediction took ', totaltime, 'seconds')

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
    '''predicts test valuesprints stats on classifier performance'''
    print("best params: ", clf.best_params_)
    print("predicts test data...")
    predict = clf.predict(X_test)
    score = get_score(predict, Y_test)
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    print('score ',score)
    return score


def get_feature(fileName):
    return genfromtxt(fileName, delimiter=',')

def get_label(fileName):
    return genfromtxt(fileName, delimiter='\n')

if __name__ == "__main__":    
    main()