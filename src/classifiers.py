from sklearn  import  metrics, preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

class support_vector:
    tuned_parameters = {"penalty":['l1', 'l2']}
    classifier = LinearSVC(max_iter=3000, dual=False)
    clf = GridSearchCV(classifier, tuned_parameters, refit=True)
    
    def fit(self, X_train, y_train):
        self.get_grid_search().fit(X_train, y_train.ravel())
    
    def get_grid_search(self):
        return self.clf
    
class baggingkneighbors:
    tuned_parameters = {'max_samples':[0.5, 1], 'max_features':[0.5, 1]}
    classifier = BaggingClassifier(KNeighborsClassifier(), n_jobs=-1)
    clf = GridSearchCV(classifier, tuned_parameters, refit=True)
    
    def fit(self, X_train, y_train):
        self.get_grid_search().fit(X_train, y_train.ravel())
    
    def get_grid_search(self):
        return self.clf
    
class kneighbors:
    tuned_parameters = {'max_samples':[0.5, 1], 'max_features':[0.5, 1]}
    classifier = BaggingClassifier(KNeighborsClassifier(), n_jobs=-1)
    clf = GridSearchCV(classifier, tuned_parameters, refit=True)
    
    def fit(self, X_train, y_train):
        self.get_grid_search().fit(X_train, y_train.ravel())
    
    def get_grid_search(self):
        return self.clf
    
    
class randomforest:
    tuned_parameters = {"criterion":["gini","entropy"], 'n_estimators':[50, 100, 150]}
    classifier = RandomForestClassifier()
    clf = GridSearchCV(classifier, tuned_parameters, refit=True)
    
    def fit(self, X_train, y_train):
        self.get_grid_search().fit(X_train, y_train.ravel())
    
    def get_grid_search(self):
        return self.clf
    
class MLP_classifier():
    tuned_parameters = {"hidden_layer_sizes":[100,200]}
    classifier = MLPClassifier(max_iter=3000)
    clf = GridSearchCV(classifier, tuned_parameters, refit=True)
    
    def fit(self, X_train, y_train):
        self.get_grid_search().fit(X_train, y_train.ravel())
    
    def get_grid_search(self):
        return self.clf
    
class decisionTree: ##er det vits å ha decision tree og random forest?
    tuned_parameters = {"criterion":["gini","entropy"], 'n_estimators':[50, 100, 150]}
    classifier = DecisionTreeClassifier()
    clf = GridSearchCV(classifier, tuned_parameters, refit=True)
    
    def fit(self, X_train, y_train):
        self.get_grid_search().fit(X_train, y_train.ravel())
    
    def get_grid_search(self):
        return self.clf


        
        

        
        
        
      