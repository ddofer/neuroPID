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

#Put a 'pickle' / try here so machine won't reload training data files each time.
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

#get the names of each protein , in order, from a multifasta file.
def FASTA_names(filename):
  try:
    f = file(filename)
  except IOError:
    print "The file, %s, does not exist" % filename
    return
  f=f.readlines()
  sequences = []
  for lines in f:
     if lines.startswith('>'):
        name = lines.rstrip('\n')
        sequences.append(name)
  print "No of fasta sequences in file: ",len(sequences)
  return sequences

def get_threshholdLocs (col_probs,thresh ):
    locs = []
    for i in range(len(col_probs)):
        if (col_probs[i]>=thresh):
            locs.append(i)
    print 'number of values exceeding threshhold: ',len(locs)
    return locs;

def Fs_names (fastas,indexes)    :
        names =[]
        for f in indexes:
#            line = fastas[f]
            names.append(fastas[f])
        return names

def write_out (Flist,orgname,methodname):
        out = open(orgname +"."+ methodname + '_bestresults.txt', "w")
        for f in Flist:
            out.write(f)
            out.write("\n")
        out.close()

if __name__ == '__main__':
    X, y = import_TrainingData()
    #Note - it's important to scale test and training data using the SAME metrics! (via a "SCALER" object)
    #(STD, etc' relative to each feature; and same adjustments/metrics for training and test features numerical range )

#TODO: Scale X Features seperately.
#    (minmax for sparse - bigrams, aa pre-suffix counts,/kmers;
#    Standardized for rest)
#    scaler = MinMaxScaler()

    scaler = StandardScaler()
    X=scaler.fit_transform(X)

    # list of text files containg organisms data,
    # they must have already been processed by our feature extractor (seperate program)
    #These contain unknown data/proteomes! (don't be confused by filenames)
#    fname_org1=r"E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Sleek_train_features\Test_OrganismsData\bombyx\Features_bombyx_.txt"
#    fname_org1=r"E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Testing_Organisms\Octopoda\Features_POS+.txt"
    fname_org1=r"D:\SkyDrive\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Testing_Organisms\c.elegans\Features_POS+.txt"
    org_FASTAS=FASTA_names(r"D:\SkyDrive\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Testing_Organisms\c.elegans\Caenorhabditis+elegansidentity1.0.fasta")
    orgname='C.elegans'
    fobject_org1= open(fname_org1, "r")
    org1= np.loadtxt(fobject_org1, delimiter="\t")
    org1= scaler.transform(org1)
#scale/standardize data by same scaling metrics used for training data

    model_rf= RandomForestClassifier(n_estimators=1200,n_jobs=-1,oob_score=True)
    model_extraTrees= ExtraTreesClassifier(n_estimators=1200,n_jobs=-1,max_features='sqrt',
                                           min_samples_leaf=1, min_samples_split=2,)
    rbf_model = SVC(C=50,kernel="rbf",gamma=5,class_weight="auto",cache_size=1100,
                    shrinking=True,probability=True)
#    linear_model=LinearSVC(C=10,penalty='l2', loss='l2', dual=False,class_weight='auto')
    linear_model=SVC(C=10,class_weight='auto', shrinking=True,probability=True,cache_size=1100)
    boost_model= GradientBoostingClassifier(n_estimators=1000,subsample=0.6,
                                            max_depth=8,learning_rate=0.05,max_features=40,)
    model_rf.fit(X,y)
    model_extraTrees.fit(X,y)
    rbf_model.fit(X,y)
    linear_model.fit(X,y)
    boost_model.fit(X,y)

    #get predictions from the machines for the data!
    print 'Probabilities now calcing..'
    extrees_prob= model_extraTrees.predict_proba(org1)
    rftrees_prob= model_rf.predict_proba(org1)
    grb_prob = boost_model.predict_proba(org1)
    rbf_prob = rbf_model.predict_proba(org1)
    preds_linear = linear_model.predict_proba(org1)

    grb_prob=grb_prob[:,1]    # Get only the ones'column probabilities
    rbf_prob=rbf_prob[:,1]
    preds_linear=preds_linear[:,1]
    extrees_prob=extrees_prob[:,1]
    rftrees_prob=rftrees_prob[:,1]

    threshhold=0.95
    #hold the index locations of the high scoring samples:
#    indexs_bestPreds_GRB = np.where(grb_prob>threshhold)
    indexs_bestPreds_RF = get_threshholdLocs(rftrees_prob,0.8)
    indexs_bestPreds_GRB = get_threshholdLocs(grb_prob,threshhold)
    indexs_bestPreds_RBF = get_threshholdLocs(rbf_prob,0.7)
    indexs_bestPreds_EXT = get_threshholdLocs(extrees_prob,0.8)
    indexs_bestPreds_SVC = get_threshholdLocs(preds_linear,threshhold)

    #extract the names of the samples from the file holding the names of each sample/protein:
    grb_names=Fs_names(org_FASTAS,indexs_bestPreds_GRB)
    rbf_names=Fs_names(org_FASTAS,indexs_bestPreds_RBF)
    ext_names=Fs_names(org_FASTAS,indexs_bestPreds_EXT)
    svc_names=Fs_names(org_FASTAS,indexs_bestPreds_SVC)
    rf__names=Fs_names(org_FASTAS,indexs_bestPreds_RF)

    write_out(grb_names,orgname,'GRB')
    write_out(rbf_names,orgname,'SVM_RBF')
    write_out(ext_names,orgname,'EXT')
    write_out(svc_names,orgname,'SVM_SVC')
    write_out(rf__names,orgname,'RF')


    #predict proba gives the probability for each sample of it belonging to each of the classes.
#    np.savetxt("Monarch_butterfly_RBF.csv", rbf_prob, delimiter=",")
#    np.savetxt("Monarch_butterfly_EXT.csv", extrees_prob, delimiter=",")
#    np.savetxt("Monarch_butterfly_RFTrees.csv", rftrees_prob, delimiter=",")
#    np.savetxt("Monarch_butterfly_GBoost.csv", grb_prob, delimiter=",")

#    np.savetxt("80_fireAnt_extratrees.txt", ext_names, delimiter="/n")
#    np.savetxt("60_fireAnt_RBF.txt", rbf_names, delimiter="/n")
#    np.savetxt("95_fireAnt _linear.txt", svc_names, delimiter="/n")
#    np.savetxt("95_fireAnt_GBoost.txt", grb_names, delimiter="/n")