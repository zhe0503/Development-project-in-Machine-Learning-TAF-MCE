import numpy as np
import pandas as pd

import sys, getopt
import cnn
from sklearn import svm

import figure

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow.keras as keras

class ml():
    def __init__(self, classifier, dataset):
        self.classifier = classifier
        self.dataset = dataset
        self.preprocess_data()

    def preprocess_data(self):
    
        def standardization(dataset, dict_col):
            # Standardization of datasets to : Gaussian with zero mean and unit variance.
            dataset[dict_col]=preprocessing.scale(dataset[dict_col])
            return dataset
            
        def labelEncoding(dataset,col):
            # code category data
            dataset[col], mapping_index = pd.Series(dataset[col]).factorize()
            return dataset[col],mapping_index

        print("preprocessing dataset ...")
            
        if len(self.dataset.columns) > 5:
            #This dataset has two types of attribute:
            col_disease_num=['id','age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
            col_disease_str=['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification']
        
            #replace missing values of the disease dataset:
            for col in col_disease_num:
                self.dataset[col]=pd.to_numeric(self.dataset[col], errors='coerce')
                self.dataset[col].fillna(self.dataset[col].median(), inplace = True)
            for col in col_disease_str:
                self.dataset[col]=self.dataset[col].astype('category')
                self.dataset[col].fillna(self.dataset[col].mode()[0], inplace = True)
                
            #Replace incorrect values:
            self.dataset['dm'] =self.dataset['dm'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'})
            self.dataset['cad'] = self.dataset['cad'].replace(to_replace = '\tno', value='no')
            self.dataset['classification'] = self.dataset['classification'].replace(to_replace = 'ckd\t', value = 'ckd')
            
            self.dataset=standardization(self.dataset, col_disease_num)
            
            for col in col_disease_str:
                self.dataset[col],_=labelEncoding(self.dataset,col)
                
        elif len(self.dataset.columns) < 26:
            self.dataset=standardization(self.dataset, self.dataset.columns[:-1])
        else:
            pass
        print("dataset preprocessed.")
    
   
        
    def train(self):
    	model_checkpoint = keras.callbacks.ModelCheckpoint('./weight.hdf5', monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    	history = self.classifier.fit(self.X_train,self.y_train,epochs=20,batch_size=512,validation_data=(self.X_test,self.y_test),callbacks=[model_checkpoint])
    	figure.draw(history)
    	pass
    	

        
    def splitTrainTest(self):
    
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.iloc[:,:-1], self.dataset.iloc[:,-1], test_size = 0.2, random_state=44 )
        
        return X_train, X_test, y_train, y_test


    def predict(self):
    	model = load_model('./weight.hdf5')
    	
    	result = model.predict(self.X_test)
    def evaluation(self):

    	scores = self.classifier.evaluate(self.X_test, self.y_test)
    	for i in range(len(scores)):
    		print("\n%s: %.2f%%" % (self.classifier.metrics_names[i], scores[i]*100))


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
    if classifier == 'CNN':
        classifier = cnn.ourCNN()
    else:
        print("Please choose a valid model.")
        
    if dataset == '' or dataset == 'banknote':
        cols_banknote=['variance of Wavelet Transformed image','skewness of Wavelet Transformed image','curtosis of Wavelet Transformed image','entropy of image','class']
        dataset=pd.read_csv('./Dataset/data_banknote_authentication.txt',names=cols_banknote)
    elif dataset == 'disease' :
        dataset=pd.read_csv('./Dataset/kidney_disease.csv',dtype=object) 
    else:
        print("Please choose a valid dataset.")
        sys.exit(2)
    #print(dataset.columns)
    model = ml(classifier, dataset)
    n_folds = 10
    model.train();
    model.predict();
    #model. evaluate();
    


