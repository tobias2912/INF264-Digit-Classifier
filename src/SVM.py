from sklearn import svm
from sklearn.metrics import accuracy_score
class SVM:
    clf = None
    def __init__(self, features, train, y):
       clf = svm.SVC()
       clf.fit(train, y)
    
    def get_accuracy(test):
        y_pred = clf.predict(test)