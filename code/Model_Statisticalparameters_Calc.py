from __future__ import division
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier
import numpy as np
from sklearn import (metrics, cross_validation)#, linear_model, preprocessing)
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numba.decorators import autojit
from sklearn import svm, preprocessing
SEED = 42  # always use a seed for randomized procedures

#==============================================================================
# User inputable files for Training data + class [0/1] labels]:
#    #fname_POS = raw_input("Enter Positive Training Set (NP+) filename & location...\n")
#         #fname_NEG = raw_input("Enter NEGative Training Set filename & location...\n")
#==============================================================================
# mean_threshold = 0.0
mean_auc = 0.0
mean_precision = 0.0
mean_recall = 0.0
mean_accuracy=0
#@autojit
def reset_means():
    global mean_auc, mean_precision, mean_recall, mean_accuracy
    mean_auc = 0.0
    mean_precision = 0.0
    mean_recall = 0.0
   # mean_threshold = 0.0
    mean_accuracy=0

#fname_NEG = "E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\AltNeg_SP_TMD\NEG_trainingset+.txt"
#fname_POS = "E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Standard\POS_NP\POS_trainingset+.txt"
@autojit
def import_TrainingData():
#    fname_NEG ="E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Standard_13\NEG\Features_NEG-.txt"
#    fname_POS ="E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Standard_13\POS_NP+_reviewed\Features_POS+.txt"
#    fname_POS ="E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Standard_13\POS_NP_all\Features_POS+.txt"
#    fname_NEG="E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Taxon_specific_negativeFilteredSets\Chordata\Features_POS+.txt"
    fname_POS=r"E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Sleek_train_features\+pos\Features_POS+.txt"
#    fname_NEG="E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Sleek_train_features\-neg\Arthropoda\Arthropoda_TMD_NEG\Features_NEG.txt"
    fname_NEG = r"E:\Dropbox\Dropbox\Books & Resources\NPID Lab BioInformatics\NPID Lab Stuff\NPID Additional Datasets\Training\Standard_13\NEG\Features_NEG-.txt"
    fileObject_POS = open(fname_POS,'r')
    test_POS = np.loadtxt(fileObject_POS, delimiter="\t")
    fileObject_NEG = open(fname_NEG,'r')
    test_NEG = np.loadtxt(fileObject_NEG, delimiter="\t")
 #For label /classification/ generation: (Classify NP+ as 1, NP- as 0..):
    samples_POS_training = len(test_POS)
    samples_NEG_training = len(test_NEG)
    y_labels = ([1] * samples_POS_training + [0] * samples_NEG_training)
    y_labels = np.asarray(y_labels) #make it a numpy array (needed for scikit )
    trainingSets = np.vstack((test_POS, test_NEG))
    return trainingSets, y_labels

#    @autojit
def CV_stats (X,y,model):
    global mean_auc, mean_precision, mean_recall, mean_accuracy
    n = 20  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 30% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.4, random_state=i*SEED)

        # train model and make predictions
        model.fit(X_train, y_train)
        preds = model.predict(X_cv)
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "( %d/%d)" % (i + 1, n)
        mean_auc += roc_auc
        accuracy=accuracy_score(y_cv, preds)
        precision = precision_score(y_cv, preds)
        recall= recall_score(y_cv, preds)
        mean_accuracy+= accuracy
        mean_precision += precision
        mean_recall += recall
        #mean_threshold += threshold
#    print "Mean accuracy: %f" % (mean_accuracy/n)
#    print "Mean precision : %f" % (mean_precision/n)
#    print "Mean recall: %f" % (mean_recall/n)
#    print "Mean AUC: %f" % (mean_auc/n)
    #print "Mean threshold: %f" % (mean_threshold/n)
    mean_accuracy=(mean_accuracy/n)
    mean_precision = mean_precision/n
    mean_recall=mean_recall/n
    mean_auc=mean_auc/n
    print  (round(mean_accuracy,3))
    print  (round(mean_precision,3))
    print (round(mean_recall,3))
    print (round(mean_auc,3))
    reset_means()

def scale_data(X):
    scaler=MinMaxScaler()
    X=scaler.fit(X).transform(X)
    return X

if __name__ == '__main__':
    X, y = import_TrainingData()
    print "Arrays and labels concatenated/imported"
    X = scale_data(X)
#    X = preprocessing.scale(X)
#TODO: Scale X Features seperately.
#    (minmax for sparse - bigrams, aa pre-suffix counts,/kmers;
#    Standardized for rest)
    model_rf= RandomForestClassifier(n_estimators=2400,n_jobs=-1,max_features="auto")
    model_extraTrees= ExtraTreesClassifier(n_estimators=2400,n_jobs=-1,max_features="auto")
    rbf_model = SVC(C=20,gamma=0.5,kernel="rbf",class_weight="auto",cache_size=4200, shrinking=True)
    linear_model=LinearSVC(C=5,penalty='l1', loss='l2', dual=False,class_weight='auto')
    boost_model= GradientBoostingClassifier(n_estimators=3200,subsample=0.7,max_depth=8,learning_rate=0.05)

    print 'Extra trees Forest stats:'
    CV_stats (X,y,model_extraTrees)
    print 'Random Forest stats:'
    CV_stats (X,y,model_rf)
    print 'RBF stats:'
    CV_stats (X,y,rbf_model)
    print 'Gradient Boosting stats:'
    CV_stats (X,y,boost_model)
    print 'Linear SVC stats:'
    CV_stats (X,y,linear_model)



