import re
from sklearn.model_selection import train_test_split
from sklearn import svm
#from sklearn.externals import joblib
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

x = []
y = []

def get_len(url):
    return len(url)

def get_url_count(url):
    if re.search('(http://)|(https://)', url, re.IGNORECASE) :
        return 1
    else:
        return 0

def get_evil_char(url):
    return len(re.findall("[<>()\'\"/]", url, re.IGNORECASE))

def get_evil_number(url):
    return len(re.findall("\d", url, re.IGNORECASE))

def get_evil_word(url):
    return len(re.findall("(alert)|(script)|(%3c)|(%3e)|(%20)|(onclick)|(onerror)|(onload)|(eval)|(src)|(prompt)|(iframe)|(style)",url,re.IGNORECASE))

def get_feature(url):
    return [get_evil_char(url),get_evil_word(url),get_url_count(url),get_len(url)]

def do_metrics(y_test,y_pred):
    print "metrics.accuracy_score:"
    print metrics.accuracy_score(y_test, y_pred)
    print "metrics.confusion_matrix:"
    print metrics.confusion_matrix(y_test, y_pred)
    print "metrics.precision_score:"
    print metrics.precision_score(y_test, y_pred)
    print "metrics.recall_score:"
    print metrics.recall_score(y_test, y_pred)
    print "metrics.f1_score:"
    print metrics.f1_score(y_test,y_pred)

def etl(filename,data,isxss):
        with open(filename) as f:
            for line in f:
                f = get_feature(line)
                data.append(f)
                if isxss:
                    y.append(1)
                else:
                    y.append(0)
        return data

etl('bad-xss-20000.txt',x,1)
etl('good-xss-20000.txt',x,0)


min_max_scaler = preprocessing.MinMaxScaler()
x_min_max=min_max_scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_min_max, y, test_size=0.2, random_state=0, stratify=y)
#grid = GridSearchCV(svm.SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=3)
#grid.fit(x_train, y_train)
#print "The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_)
#print grid.best_params_["C"]


#clf = svm.SVC(kernel='rbf', C=grid.best_params_["C"], gamma=grid.best_params_["gamma"]).fit(x_train, y_train)
clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
#clf = svm.SVC(kernel='rbf', C=1, probability=True).fit(x_train, y_train)
#probas_ = clf.predict_proba(x_test)
#fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
#roc_auc = auc(fpr, tpr)
#plt.plot(fpr, tpr, lw=1, label='ROC(area = %0.2f)' % (roc_auc))
#
#
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()

sv=clf.support_vectors_
#print "support_vectors:"
#print sv
print "support_vectors_shape:"
print sv.shape

y_pred = clf.predict(x_test)
do_metrics(y_test, y_pred)

#joblib.dump(clf,"xss-svm-data-module.m")


