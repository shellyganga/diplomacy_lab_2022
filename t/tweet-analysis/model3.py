import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
#import the necessary libraries for dataset preparation, feature engineering, model training
from sklearn import model_selection, preprocessing, metrics, linear_model, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import (RandomUnderSampler,
                                    NearMiss,
                                    InstanceHardnessThreshold,
                                    CondensedNearestNeighbour,
                                    EditedNearestNeighbours,
                                    RepeatedEditedNearestNeighbours,
                                    AllKNN,
                                    NeighbourhoodCleaningRule,
                                    OneSidedSelection,
                                    TomekLinks)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
import pandas as pd, numpy, string
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer

#Remove Special Charactors
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer



from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
import pandas as pd


df = pd.read_csv("/Users/shellyschwartz/Downloads/webis-clickbait-16/truth/data2.csv")
df = df.dropna()
train_x, valid_x, train_y, valid_y = train_test_split(df['text'], df['clickbait'], test_size=0.33, random_state=42)
print(train_y)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
print(train_y)
valid_y = encoder.fit_transform(valid_y)



# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000)
tfidf_vect.fit(df['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


# Return the f1 Score
def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(valid_y, predictions), classifier







accuracyORIGINALlr = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),xtrain_tfidf, train_y, xvalid_tfidf)[0]
#print ("LR Baseline, WordLevel TFIDF: ", accuracyORIGINALlr)
accuracyORIGINALsvm = train_model(svm.LinearSVC(), xtrain_tfidf, train_y, xvalid_tfidf)[0]
#print ("SVM Baseline, WordLevel TFIDF: ", accuracyORIGINALsvm)



# from sklearn.metrics import confusion_matrix
# pred = train_model(svm.LinearSVC(), xtrain_tfidf, train_y, xvalid_tfidf)[0]
# matrix = confusion_matrix(valid_y, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))


#SMOTE
sm = SMOTE(random_state=777)
sm_xtrain_tfidf, sm_train_y = sm.fit_resample(xtrain_tfidf, train_y)
accuracySMOTElr = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),sm_xtrain_tfidf, sm_train_y, xvalid_tfidf)[0]
#print ("LR SMOTE, WordLevel TFIDF: ", accuracySMOTElr)
accuracySMOTEsvm = train_model(svm.LinearSVC(), sm_xtrain_tfidf, sm_train_y, xvalid_tfidf)[0]
#print ("SVC SMOTE, WordLevel TFIDF: ", accuracySMOTEsvm)


#Borderline SMOTE
bsm = BorderlineSMOTE()
bsm_xtrain_tfidf, bsm_train_y = bsm.fit_resample(xtrain_tfidf, train_y)
accuracyBSMOTElr = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),bsm_xtrain_tfidf, bsm_train_y, xvalid_tfidf)[0]
#print ("LR Borderline SMOTE, WordLevel TFIDF: ", accuracyBSMOTElr)
accuracyBSMOTEsvm = train_model(svm.LinearSVC(),bsm_xtrain_tfidf, bsm_train_y, xvalid_tfidf)[0]
# print ("SVM Borderline SMOTE, WordLevel TFIDF: ", accuracyBSMOTEsvm)


best_acc = max(accuracyBSMOTElr,accuracyBSMOTEsvm, accuracySMOTElr, accuracySMOTEsvm, accuracyORIGINALlr, accuracyORIGINALsvm)



if(best_acc == accuracyBSMOTElr):
    model = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),bsm_xtrain_tfidf, bsm_train_y, xvalid_tfidf)[1]
    joblib.dump(model, "/Users/shellyschwartz/PycharmProjects/diplomacy_lab/finalized_model.sav")
elif(best_acc == accuracyBSMOTEsvm):
    model = train_model(svm.LinearSVC(), bsm_xtrain_tfidf, bsm_train_y, xvalid_tfidf)[1]
    joblib.dump(model, "/Users/shellyschwartz/PycharmProjects/diplomacy_lab/finalized_model.sav")
elif(best_acc == accuracySMOTElr):
    model = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),sm_xtrain_tfidf, sm_train_y, xvalid_tfidf)[1]
    joblib.dump(model, "/Users/shellyschwartz/PycharmProjects/diplomacy_lab/finalized_model.sav")
elif(best_acc == accuracySMOTEsvm):
    model = accuracySMOTEsvm = train_model(svm.LinearSVC(), sm_xtrain_tfidf, sm_train_y, xvalid_tfidf)[1]
    joblib.dump(model, "/Users/shellyschwartz/PycharmProjects/diplomacy_lab/finalized_model.sav")
elif(best_acc == accuracyORIGINALlr):
    model =  train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),xtrain_tfidf, train_y, xvalid_tfidf)[1]
    joblib.dump(model, "/Users/shellyschwartz/PycharmProjects/diplomacy_lab/finalized_model.sav")
elif(best_acc == accuracyORIGINALsvm):
    model = train_model(svm.LinearSVC(), xtrain_tfidf, train_y, xvalid_tfidf)[1]
    joblib.dump(model, "/Users/shellyschwartz/PycharmProjects/diplomacy_lab/finalized_model.sav")