import numpy as np
import pandas as pd
import sys, getopt

class ml():
    def __init__(self, classifier, dataset):
        self.classifier = classifier
        self.dataset = dataset

    def preprocess_data(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def validate(self):
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
if __name__ == '__main__':
    argv_test(sys.argv[1:])
