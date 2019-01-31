#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_excel("cancer.xlsx")


# In[3]:


df.info()


# In[4]:


df.head(10000)


# In[5]:


df.shape


# # Preprocessing

# In[6]:


import numpy as np
df.replace('?',np.NaN,inplace=True)
df.replace('inactive',0,inplace=True)
df.replace('active',1,inplace=True)


# In[7]:


df.info()


# In[8]:


df.fillna(df.mean(), inplace = True)


# In[31]:


y=df['inactive']
X=df[list(df.select_dtypes(include=['float64']))]
print(X)


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[116]:


X_test


# In[117]:


from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)


# # SGD Classifier

# In[121]:


import gc
from sklearn.metrics import mean_squared_error
from sklearn.linear_model.stochastic_gradient import SGDClassifier

estimator = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, 
                          max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                          learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False, 
                          n_iter=None)
mean_squared_error(y_test,estimator.fit(X_train,y_train).predict(X_test))


# In[46]:


from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


pca = PCA(n_components=175)
pca.fit(X)
X_pca = pca.transform(X)


# In[47]:


X_pca.shape


# # LassoCV

# In[36]:


import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
clf = LassoCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf)
sfm.fit(X, y)
n_features = sfm.transform(X)


# In[37]:


n_features.shape


# In[58]:


clf.fit(n_features,y)
clf.score(n_features, y)


# # lassocv and pca

# In[48]:


from sklearn.pipeline import Pipeline

# fits PCA, transforms data and fits the decision tree classifier
# on the transformed data
pipe = Pipeline([('pca', PCA()),
                 ('tree', DecisionTreeClassifier())])

pipe.fit(X_pca, y)

pipe.predict(X_pca)


# In[49]:


from sklearn.cross_validation import cross_val_score
val_score_pca = cross_val_score(pipe,X_pca, y)
print(val_score_pca)


# In[66]:


plot_learning_curve(LassoCV(), "Learning curve (LassoCV)", n_features, y, (0, 0.5), cv=cv, n_jobs=4)

plt.show()


# In[50]:


from sklearn.cross_validation import cross_val_score
val_score_lasso = cross_val_score(pipe,n_features, y)
print(val_score_lasso)


# In[ ]:


plot_learning_curve(clf, "Learning curve (PCA)", X_train_1, y_train_1, (0.95, 1.01), cv=cv, n_jobs=4)

plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_pca, y, test_size=0.33, random_state=42)


# In[16]:


from sklearn import svm

clf_svm=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf_svm.fit(X_train_1,y_train_1)
y_pred = clf_svm.predict(X_test_1)

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_1, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
plot_confusion_matrix(cm,[0,1],normalize=False)



# In[17]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test_1,y_pred))


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[22]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
title = "Learning Curves (Kernel-SVM)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = clf_svm
plot_learning_curve(estimator, title, X_train_1, y_train_1, (0.95, 1.01), cv=cv, n_jobs=4)

plt.show()


# In[23]:


from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state = 0)
clf = clf.fit(X_train_1, y_train_1)
y_pred = clf.predict_proba(X_test_1)


# In[24]:


from sklearn.model_selection import cross_val_score

cross_val_score(clf, X_train_1, y_train_1, cv=10)


# In[25]:


plot_learning_curve(clf, "Learning curve (Decision-Tree)", X_train_1, y_train_1, (0.95, 1.01), cv=cv, n_jobs=4)

plt.show()


# In[33]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20

classifiers = [
    ("SGD", SGDClassifier()),
    ("ASGD", SGDClassifier(average=True)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
]

xx = 1. - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()


# In[67]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[71]:


classifiers = [
    ("SGD", SGDClassifier()),
    ("ASGD", SGDClassifier(average=True)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0)),
    ("Logistic Regression", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
]

for name, clf in classifiers:
    title = name
# SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = clf
    plot_learning_curve(clf, title, X_train, y_train, (0.9, 1.01), cv=cv, n_jobs=4)

    plt.show()


# In[78]:


title = "Logistic Regression"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
clf=LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0])
plot_learning_curve(clf, title, X_train, y_train, (0.99,1.00), cv=cv, n_jobs=4)

plt.show()


# In[ ]:




