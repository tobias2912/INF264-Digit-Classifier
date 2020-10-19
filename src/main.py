import pandas as pd
from sklearn  import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

def get_feature(fileName):
    return genfromtxt(fileName, delimiter=',')

def get_label(fileName):
    return genfromtxt(fileName, delimiter='\n')

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
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(digits, label,      
                                                                test_size=0.3, 
                                                                shuffle=True, 
                                                                random_state=seed)
    seed = 77
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, 
                                                                test_size=0.5, 
                                                                shuffle=True, 
                                                                random_state=seed)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
       
def svc_classifier(X_train, Y_train, X_test, Y_test):
    '''Create a SVC model'''
    classifier = svm.SVC(gamma = 0.001)
    classifier.fit(X_train, Y_train)
    predict = classifier.predict(X_test)
    score = classifier.score(predict, Y_test)
    print('score ',score)
   
def plot(digits, label, predict):
    '''Plot a single image and print the label to console'''
    img = digits.reshape(digits.shape[0], 28, 28)
    plt.imshow(img[predict], cmap="Greys")
    print(label[predict])
    plt.show()    
    
if __name__ == "__main__":    
    
    if False:
        digits = get_feature('../data/handwritten_digits_images.csv')
        label = get_label('../data/handwritten_digits_labels.csv')
        create_smaller_file(digits, label)
    else:
        digits=get_feature('../data/digit_smaller.csv')
        label=get_feature('../data/digit_smaller.csv')
    
    #plot(digits, label, 3)
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split(digits, label)
    
    svc_classifier(X_train, Y_train, X_test, Y_test)
    
        
        
        
    



    
     

    

    
