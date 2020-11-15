import pandas as pd
import sys, getopt

#classifier
from sklearn import svm
from xgboost.sklearn import XGBClassifier
from cnn import ourCNN

#utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

class ml_model():
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
            col_disease_num=['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
            col_disease_str=['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification']
            del self.dataset['id']
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
        self.classifier.fit(self.X_train, self.y_train)
        
    def splitTrainTest(self):
        print("Data splitting ... ")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.iloc[:,:-1], self.dataset.iloc[:,-1], test_size = 0.2, random_state=44 )
        print("Split finished : Training size : %d, Test size : %d" %(self.X_train.shape[0], self.X_test.shape[0]))

    def predict(self):
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)

    def evaluation(self):
        pass




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
    if classifier == '' or classifier == "svm":
        classifier = svm.SVC(kernel='linear')
    elif classifier == 'xgboost':
        classifier = XGBClassifier(learning_rate= 0.2, max_depth= 7,objective='binary:logistic',n_estimators= 100,gamma=0.5,scale_pos_weight=3, n_jobs= -1,reg_alpha=0.2,reg_lambda=1,random_state =1367)
    elif classifier == 'CNN':
        classifier = ourCNN()
    else:
        print("Please choose a valid classifier : \n<svm> for SVM(by default) \n <xgboost> for XGBoost \n <CNN> for a 3-layer deep-learning model")
        sys.exit(2)

    if dataset == 'banknote' or dataset == '':
        cols_banknote=['variance of Wavelet Transformed image','skewness of Wavelet Transformed image','curtosis of Wavelet Transformed image','entropy of image','class']
        dataset=pd.read_csv('./Dataset/data_banknote_authentication.txt',names=cols_banknote)
    elif dataset == 'disease':
        dataset=pd.read_csv('./Dataset/kidney_disease.csv', dtype=object,)
    else:
        print("Please choose a valid dataset : \n<banknote> for banknote authentication dataset(by default) \n and <disease> for Chronic KIdney Disease dataset")
        sys.exit(2)

    model = ml_model(classifier, dataset)
    model.train()
    model.predict()
    
    


