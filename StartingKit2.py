#!/Users/mac/anaconda/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from fuzzywuzzy import fuzz


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.learning_curve import learning_curve,validation_curve

DATA_PATH ="."              # PATH to your files
REPORT = 0                  # 0 --> no report regarding columns, 1 --> report
SHOW_PLOTS_UNIVARIATE = 1   # 0 --> no plots regarding columns, 1 --> plots
SHOW_CURVES = 0             # 0 --> no curves regarding models, 1 --> curves
DO_STUDY = 0                # 0 --> no strudy on different models, 1 --> study
class ParseMetadata(object):
    """Parsing Metadata file """
    def __init__(self,filename):
        self.filename = filename
    
    def parse_metadata_file(self):
        with open(os.path.join(DATA_PATH,self.filename),'r')  as f:
            metadatas = f.readlines()
        f.close()
        # Keep only lines from 24 to 69 for the headers
        headerdata = metadatas[23:68]
        # Keep only lines from 81 to 121 for the other informations
        infodata = metadatas[81:121]
        return (headerdata,infodata) 

    def build_headers(self,headerdata):
        headers = {}
        for data in headerdata:
            key = str.lower(re.split(r'\t',data)[0].strip("\n| "))
            value = re.split(r'\t',data)[-1].strip("\n")
            headers[key] = value
        return headers

    def build_infos(self,infodata):
        infos = []
        for data in infodata:
            infos.append(data.split('(', 1)[1].split(')')[0])
        return infos
    
    
def plot_learning_curve(estimator, X, y, ylim=(0, 1.1), cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None):
    plt.title("Learning curves for %s" % type(estimator).__name__)
    plt.ylim(*ylim); plt.grid()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    # Check SHOW_CURVES parameter
    if SHOW_CURVES != 0: plt.show()
    print("Best test score: {:.4f}".format(test_scores_mean[-1]))
    return (estimator, test_scores_mean[-1])
    
def plot_validation_curve(estimator, X, y, param_name, param_range,
                          ylim=(0, 1.1), cv=5, n_jobs=-1, scoring=None):
    estimator_name = type(estimator).__name__
    plt.title("Validation curves for %s on %s"
              % (param_name, estimator_name))
    plt.ylim(*ylim); plt.grid()
    plt.xlim(min(param_range), max(param_range))
    plt.xlabel(param_name)
    plt.ylabel("Score")

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name, param_range,
        cv=cv, n_jobs=n_jobs, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.semilogx(param_range, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.semilogx(param_range, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()
    print("Best test score: {:.4f}".format(test_scores_mean[-1]))
    
def do_all_study(X,y):
    
    names = [ "Decision Tree","Gradient Boosting",
             "Random Forest", "AdaBoost", "Naive Bayes"]

    classifiers = [
        #SVC(),
        DecisionTreeClassifier(max_depth=10),
        GradientBoostingClassifier(max_depth=10, n_estimators=20, max_features=1),
        RandomForestClassifier(max_depth=10, n_estimators=20, max_features=1),
        AdaBoostClassifier()]
    for name, clf in zip(names, classifiers):
        estimator,score = plot_learning_curve(clf, X_train, y_train, scoring='roc_auc')


    clf_GBC = GradientBoostingClassifier(max_depth=10, n_estimators=20, max_features=1)
    param_name = 'n_estimators'
    param_range = [1, 5, 10, 20,40]

    plot_validation_curve(clf_GBC, X_train, y_train,
                          param_name, param_range, scoring='roc_auc')
    clf_GBC.fit(X_train,y_train)
    y_pred_GBC = clf_GBC.predict_proba(X_test)[:,1]
    print("ROC AUC GradientBoostingClassifier: %0.4f" % roc_auc_score(y_test, y_pred_GBC))

    clf_AB = AdaBoostClassifier()
    param_name = 'n_estimators'
    param_range = [1, 5, 10, 20,40]

    plot_validation_curve(clf_AB, X_train, y_train,
                          param_name, param_range, scoring='roc_auc')
    clf_AB.fit(X_train,y_train)
    y_pred_AB = clf_AB.predict_proba(X_test)[:,1]
    print("ROC AUC AdaBoost: %0.4f" % roc_auc_score(y_test, y_pred_AB))


# ********  MAIN ********
metafile = ParseMetadata("census_income_metadata.txt")
headerdatas,infosdata = metafile.parse_metadata_file()

infos = metafile.build_infos(infosdata)
headers = metafile.build_headers(headerdatas)

# Add by hand most problematic headers
headers["detailed industry recode"] = "ADTIND"
headers["detailed occupation recode"] = "ADTOCC"
headers["race"] = "ARACE"
headers["enroll in edu inst last wk"] = "AHSCOL"
headers["marital stat"] ="AMARITL"


# Creating colnames and descript lists as the ACRONYM and DESCRIPTION of each column
# I am using fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy)
colnames =[]
descript =[]
count = 0
for item in infos:
    for head in headers.keys():
       if (fuzz.ratio(item,head) > 92) & (headers[head] not in colnames):
          colnames.append(headers[head])
          descript.append(head)

colnames.append("YEAR")
colnames.append("SAVINGS")


# Import learning data 
train_df = pd.read_csv( "census_income_learn.csv",delimiter=',')

# Drop instance weight column (24)
train_df.drop([train_df.columns[24]], axis=1, inplace=True)
# Put column names in columns
train_df.columns = colnames

#Factorize learning data
data_encoded = train_df.apply(lambda x: pd.factorize(x)[0])
if REPORT != 0:
    for item in data_encoded:
        print "*** COLUMN NAME ***",item
        print "*** DTYPE ***",train_df[item].dtype
        if (train_df[item].dtype == "int64"):
            print "*** MIN VALUE ***",np.min(train_df[item])
            print "*** MAX VALUE ***",np.max(train_df[item])
            print "*** Variance ***",np.var(train_df[item]),"\n"
            if SHOW_PLOTS_UNIVARIATE != 0:
                train_df.hist(column=item)
                plt.axvline(x = train_df[item].mean(), linewidth = 2, color = 'r')
                plt.show()
        else:
            train_df[item] = train_df[item].astype('category')
            print "*** Value counts ***",train_df[item].value_counts(),"\n"
            if SHOW_PLOTS_UNIVARIATE != 0:
                ordering = np.argsort(train_df[item].value_counts(sort = False))[::-1]
                #print train_df[item].cat.categories[ordering]
                x = np.arange(len(train_df[item].cat.categories))
                plt.bar(x, train_df[item].value_counts())
                plt.xticks(x + 0.5, train_df[item].cat.categories[ordering], rotation=90, fontsize=10)
                plt.title(item)
                plt.show()
   
X = data_encoded.drop("SAVINGS",axis=1).values
y = data_encoded["SAVINGS"].values


X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.4, random_state=0)

# Check DO_STUDY parameter
if DO_STUDY != 0:
    do_all_study(X_train,y_train)

clf = AdaBoostClassifier()
clf.fit(X_train,y_train)
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)


# Import test files
test_df = pd.read_csv( "census_income_test.csv",delimiter=',')
# Drop instance weight column (24)
test_df.drop([test_df.columns[24]], axis=1, inplace=True)

test_df.columns = colnames

test_encoded = test_df.apply(lambda x: pd.factorize(x)[0])

X_t = test_encoded.drop("SAVINGS",axis=1).values
y_t = test_encoded["SAVINGS"].values

final_pred = clf.predict(X_t)

plt.figure(figsize=(10, 5))

ordering = np.argsort(clf.feature_importances_)[::-1]

importances = clf.feature_importances_[ordering]
feature_names = test_df.columns[ordering]

y_pred_F = clf.predict_proba(X_t)[:,1]
print("Finale ROC AUC : %0.4f" % roc_auc_score(y_t, y_pred_F))

x = np.arange(10)
plt.bar(x, importances[:10])
plt.xticks(x + 0.5, feature_names[:10], rotation=90, fontsize=10)
plt.show()