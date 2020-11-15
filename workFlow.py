import numpy as np
import pandas as pd
import sys, getopt
from sklearn import svm

import cnn.CNN
import figure.draw
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

class ml():
    def __init__(self, classifier, dataset):
        self.classifier = classifier
        self.dataset = dataset
        self.preprocess_data()

    def preprocess_data(self):
        
		
        pass

    def fit(self):
    	model_checkpoint = keras.callbacks.ModelCheckpoint('./weight/weight_cnn.hdf5', monitor="val_loss", mode="min", verbose=1, save_best_only=True)

        classifier = CNN.fit(x_train,y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val),callbacks=[model_checkpoint])
        
        draw(classifier)

    def predict(self):
        pass

    def evaluation(self):
		scores = CNN.evaluate(x_test, y_test)
		for i in range(len(scores)):
		 print("\n%s: %.2f%%" % (CNN.metrics_names[i], scores[i]*100))
        pass


def argv_test(argv):
    dataset = ''
    classifier = ''
    try:
        opts, args = getopt.getopt(argv, "hd:c:", ["dataset=", "classifier="])
    except getopt.GetoptError as e:
        #print(e)
        print("python workFlow.py -d <dataset> -c <classifier>")
        print("Please choose dataset from Banknote Authentication Dataset and Chronic Kidney Disease")
        print("Classifier option : xxx ; svm used by default")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("workFlow.py -d <dataset> -c <classifier>")
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-c", "--classifier"):
            classifier = arg
    print("dataset chose : ", dataset)
    print("classifier chose : ", classifier)
    return dataset, classifier
if __name__ == '__main__':
    dataset, classifier = argv_test(sys.argv[1:])
    if classifier == '':
        classifier = svm.SVC(kernel='linear')
    model = ml(dataset, classifier)

