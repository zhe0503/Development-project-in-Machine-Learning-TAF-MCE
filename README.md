# Development project in Machine Learning
TAF MCE


Group 1: LI Zheng, WEI Jiahui, WU Zhe


Option 1:  Binary Classification

Banknote Authentication Dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authenticationIChronic 

Kidney Disease: https://www.kaggle.com/mansoordaku/ckdiseaseI

## Description
This project contains in total 3 maiching learning methods to 
solve the binary classification problems : one for the Banknote authentication dataset, the other Kidney Disease dataset.

We use three methods : SVM, XGBoost and a 3-layer Neural Network and compare its results in dealing with different datasets.

## Requirements

In order to successfully run this project in your end, make sure you use 
    Python >= 3.5

To install the required package, use : 
        
        pip install requirements.txt

## Project Structure


```
Project
└── workFlow.py
└── dnn.py
└── pre-processing-dataset.ipynb
└── Dataset
    └── data_banknote_authentication.txt
    └── kedney_disease.csv
```

workFlow.py contains the whole workFlow to use our three methods.

dnn.py contains our 3-layer neural network model and related functions needed to complete this methode.

pro-processing-kedney_disease.ipynb is a notebook to visually present our preprocessing methods for these two datasets.

## Usage

Use the command below to see the usage manual : 
        
        python workFlow.py -h 
 
Usage : 
        
        python workFlow.py -c <classifier> -d <dataset>
    
where classifier is the classifier, it could be svm for a SVM classifier, xgboost for XGBoost classifier and DNN for a 3-layer NN model. Default for SVM.

And dataset should be one of these two values : "banknote" for Banknote Authentication Dataset and "disease" for Kidney Disease. Default for Banknote Authentication Dataset.
