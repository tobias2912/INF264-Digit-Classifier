

class classifiers:
    def svc_classifier(X_train, Y_train, X_test, Y_test):
        '''Create a SVC model'''
        Y_train = Y_train.reshape(-1,1)
        # Y_train = Y_train.ravel()
        Y_test = Y_test.reshape(-1,1)
        print('*******SHAPE of  Y: ', Y_train.shape, Y_test.shape)
        # classifier = svm.SVC(gamma = 0.001)
        classifier = KNeighborsClassifier(n_neighbors=3)
        print("fits classifier...")
        classifier.fit(X_train, Y_train.ravel())
        print("predicts test data...")
        predict = classifier.predict(X_test)
        # score = classifier.score(predict, Y_test)
        score = get_score(predict, Y_test)
        disp = metrics.plot_confusion_matrix(classifier, X_test, Y_test)
        disp.figure_.suptitle("Confusion Matrix")
        print("Confusion matrix:\n%s" % disp.confusion_matrix)
        print('score ',score)
        
        
        
      