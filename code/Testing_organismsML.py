# -*- coding: utf-8 -*-
#! E:\Python27

from __future__ import division
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier
import numpy as np
from sklearn import (metrics, cross_validation)#, linear_model, preprocessing)
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.preprocessing import MinMaxScaler
from numba.decorators import autojit
from sklearn import svm, preprocessing
from sklearn.preprocessing import StandardScaler

#not needed to Put a 'pickle' / try here so machine won't reload training data files each time. [ only for the Trained machine]
#(same for matrixes and model of trained machine later in the program.)
@autojit
def import_TrainingData():
#    fname_POS="E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Sleek_train_features\+pos\Features_POS+.txt"
#    fname_NEG = "E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Standard_13\NEG\Features_NEG-.txt"
    fname_NEG = r"D:\SkyDrive\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Standard_13\NEG\Features_NEG-.txt"
    fname_POS=r"D:\SkyDrive\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Standard_13\POS_NP1.0_all\Features_POS+.txt"
#    fname_POS="E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Sleek_train_features\Test_OrganismsData\_Sets Training_filtered\c.elegans\Features_POS+.txt"
    fileObject_POS = open(fname_POS, "r")
    test_POS = np.loadtxt(fileObject_POS, delimiter="\t")
    fileObject_NEG = open(fname_NEG, "r")
    test_NEG = np.loadtxt(fileObject_NEG, delimiter="\t")
 #For label /classification/ generation: (Classify NP+ as 1, NP- as 0..):
    samples_POS_training = len(test_POS)
    samples_NEG_training = len(test_NEG)
    y_labels = ([1] * samples_POS_training + [0] * samples_NEG_training)
    y_labels = np.asarray(y_labels) #make it a numpy array (needed for scikit )
    trainingSets = np.vstack((test_POS, test_NEG))
    return trainingSets, y_labels

if __name__ == '__main__':
    X, y = import_TrainingData()
    #Note - it's important to scale test and training data using the SAME "SCALE"
    #(STD, etc' relative to each feature; and same adjustments/metrics for training and test features numerical range )

#TODO: Scale X Features seperately.
#    (minmax for sparse - bigrams, aa pre-suffix counts,/kmers;
#    Standardized for rest)
#    scaler = MinMaxScaler()
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    #This can easily be turned into a def or made to take raw user input/filepathname as input for the file's location
    # list of text files containg organisms data,
    # they must have already been processed by our feature extractor (seperate program)
    #These contain unknown data/proteomes! (don't be confused by filenames)
#    fname_org1=r'E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Sleek_train_features\Test_OrganismsData\M Butterfly\Features_POS+.txt'
#    fname_org1="E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Sleek_train_features\Test_OrganismsData\APIS 7460\Features_POS+.txt"
#    fname_org1=r"E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Sleek_train_features\Test_OrganismsData\bombyx\Features_bombyx_.txt"
#    fname_org1=r"E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Testing_Organisms\Octopoda\Features_POS+.txt"
    fname_org1=r"D:\SkyDrive\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Testing_Organisms\Octopoda\Features_POS+.txt"
    fobject_org1= open(fname_org1, "r")
    org1= np.loadtxt(fobject_org1, delimiter="\t")
    org1= scaler.transform(org1)

#scale/standardize data by same scaling metrics used for training data

#    model_rf= RandomForestClassifier(n_estimators=200,n_jobs=-1,max_features='sqrt',oob_score=True)
    model_extraTrees= ExtraTreesClassifier(n_estimators=1000,n_jobs=-1,max_features='sqrt',
                                           min_samples_leaf=1, min_samples_split=2,)
    rbf_model = SVC(C=20,kernel="rbf",gamma=0.5,class_weight="auto",cache_size=4200,
                    shrinking=True,probability=True)
    linear_model=LinearSVC(C=10,penalty='l2', loss='l2', dual=False,class_weight='auto')
#    linear_model=SVC(C=10,class_weight='auto', shrinking=True,probability=True,cache_size=5600)
    boost_model= GradientBoostingClassifier(n_estimators=1000,subsample=0.5,
                                            max_depth=8,learning_rate=0.05,max_features=40,)

   #Train each model on the labelled data
# implement a pickle here after training! # http://scikit-learn.org/dev/tutorial/basic/tutorial.html#model-persistence

#"Try", prior to training
#    model_rf.fit(X,y)
    model_extraTrees.fit(X,y)
    rbf_model.fit(X,y)
    linear_model.fit(X,y)
    boost_model.fit(X,y)

#Pickle the TRAINED (post fitting) machines!
    #get predictions from the machines for the data!

#    preds_rf = model_rf.predict(org1)
#    print "RF",sum(preds_rf)
    preds_ext = model_extraTrees.predict(org1)
    #Also support predict+probability
    print "ExtraTrees",sum(preds_ext)
    preds_rbf = rbf_model.predict(org1)
    #predict_proba - Predict PROBABILITIES (as opposed to binary 0/1 simplified)
    print "RBF",sum(preds_rbf)
    preds_linear = linear_model.predict(org1)
    print "Linear",sum(preds_linear)
    preds_grb = boost_model.predict(org1)
    print "GR Boost",sum(preds_grb)
    print 'Probabilities now calcing..'
    extrees_prob= model_extraTrees.predict_proba(org1)
#    print extrees_prob
#    rftrees_prob= model_rf.predict_proba(org1)
    grb_prob = boost_model.predict_proba(org1)
    rbf_prob = rbf_model.predict_proba(org1)
    print 'last - ',rbf_prob
#    preds_linear = linear_model.predict_proba(org1)
    #predict proba gives the probability for each sample of it belonging to each of the classes.
    np.savetxt("ARTHT_Trained_fireAnt_extratrees.csv", extrees_prob, delimiter=",")
    np.savetxt("ARTHT_Trained_fireAnt_RBF.csv", rbf_prob, delimiter=",")
#    np.savetxt("ARTHT_Trained_fireAnt _RFTrees.csv", rftrees_prob, delimiter=",")
    np.savetxt("ARTHT_Trained_fireAnt_GBoost.csv", grb_prob, delimiter=",")