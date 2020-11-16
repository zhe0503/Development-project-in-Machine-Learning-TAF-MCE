import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt
import statistics 
import matplotlib.pyplot as plt

#classifier
from sklearn import svm
from xgboost.sklearn import XGBClassifier
from dnn import ourDNN

#utils
from itertools import *
from sklearn.metrics import make_scorer,precision_score,recall_score
from sklearn.model_selection import train_test_split,cross_validate
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from dnn import draw
import tensorflow.keras as keras
from keras.models import load_model


class ml_model():
    def __init__(self, classifier, dataset, option):
        self.classifier = classifier
        self.dataset = dataset
        self.preprocess_data()
        self.option = option

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
            col_disease_num=['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
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
            print("Invalid dataset.")
            sys.exit(2)

        self.splitTrainTest()
        print("dataset preprocessed.")

    def train(self):
        if self.option == "ml":
            self.classifier.fit(self.X_train, self.y_train)
        else:
            model_checkpoint = keras.callbacks.ModelCheckpoint('./weight.hdf5', monitor="val_loss", mode="min",
                                                               verbose=1, save_best_only=True)
            history = self.classifier.fit(self.X_train, self.y_train, epochs=20, batch_size=64,
                                          validation_data=(self.X_test, self.y_test), callbacks=[model_checkpoint])
            draw(history)

    def splitTrainTest(self):

        print("Data splitting ... ")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.iloc[:,:-1], self.dataset.iloc[:,-1], test_size = 0.2, random_state=44 )
        print("Split finished : Training size : %d, Test size : %d" %(self.X_train.shape[0], self.X_test.shape[0]))

    def predict(self):
        if self.option != "ml":
            self.classifier = load_model('./weight.hdf5')
            y_pred = (self.classifier.predict(self.X_test) > 0.5).astype("int32")
        else:
            y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        if len(self.dataset.columns) > 5:

            classes = ['ckd', 'notckd']
            self._plot_confusion_matrix(cm, classes, normalize=True)
        else:
            self._plot_confusion_matrix(cm, ['0', '1'], normalize=True)



    def crossValidation(self):
        if option == "ml":
            scoring = {'accuracy': 'accuracy',
                        'precision' : make_scorer(precision_score, average = 'micro'),
                        'recall' : make_scorer(recall_score, average = 'micro'), }
            cross_val_scores = cross_validate(self.classifier, self.dataset.iloc[:,:-1], self.dataset.iloc[:,-1], cv=10, scoring=scoring)
            print('mean precision for 10-crossed validation:',statistics.mean(cross_val_scores['test_precision']))
            print('mean recall for 10-crossed validation:',statistics.mean(cross_val_scores['test_recall']))
            print('mean accuracy for 10-crossed validation:',statistics.mean(cross_val_scores['test_accuracy']))

    def _plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize = 20)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label', fontsize = 20)
        plt.xlabel('Predicted label', fontsize = 20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.show()



def argv_test(argv):
    dataset = ''
    classifier = ''
    try:
        opts, args = getopt.getopt(argv, "hd:c:", ["dataset=", "classifier="])
    except getopt.GetoptError:
        print("use : python workFlow.py -d <dataset> -c <classifier>")
        print("Please choose a valid classifier : \n<svm> for SVM(by default) \n <xgboost> for XGBoost \n <CNN> for a 3-layer deep-learning model")
        print("Please choose a valid dataset : \n<banknote> for banknote authentication dataset(by default) \n and <disease> for Chronic KIdney Disease dataset")
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
    option = 'ml'

    if dataset == 'banknote' or dataset == '':
        cols_banknote=['variance of Wavelet Transformed image','skewness of Wavelet Transformed image','curtosis of Wavelet Transformed image','entropy of image','class']
        dataset=pd.read_csv('./Dataset/data_banknote_authentication.txt',names=cols_banknote)
    elif dataset == 'disease':
        dataset=pd.read_csv('./Dataset/kidney_disease.csv', dtype=object)
        del dataset['id']
    else:
        print("Please choose a valid dataset : \n<banknote> for banknote authentication dataset(by default) \n and <disease> for Chronic KIdney Disease dataset")
        sys.exit(2)

    if classifier == '' or classifier == "svm":
        classifier = svm.SVC(kernel='linear')
    elif classifier == 'xgboost':
        classifier = XGBClassifier(learning_rate= 0.2, max_depth= 7,objective='binary:logistic',n_estimators= 100,gamma=0.5,scale_pos_weight=3, n_jobs= -1,reg_alpha=0.2,reg_lambda=1,random_state =1367)
    elif classifier == 'DNN':
        classifier = ourDNN(len(dataset.columns) - 1)
        option = 'dl'
    else:
        print("Please choose a valid classifier : \n<svm> for SVM(by default) \n <xgboost> for XGBoost \n <DNN> for a 3-layer deep-learning model")
        sys.exit(2)

    model = ml_model(classifier, dataset, option)
    model.train()
    model.predict()
    model.crossValidation()
