#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_excel("cancer.xlsx")


# In[2]:


df.info()


# In[3]:


df.head(10000)


# In[125]:


df.shape


# In[126]:


import numpy as np
df.replace('?',np.NaN,inplace=True)
df.replace('inactive',0,inplace=True)
df.replace('active',1,inplace=True)


# In[6]:


df.info()


# In[143]:


df.fillna(np.mean(df), inplace = True)


# In[251]:


y=df['inactive']
X=df[list(df.select_dtypes(include=['float64']))]
# print(y)


# In[278]:


import matplotlib.pyplot as plt
plt.plot(df.iloc[0:21799,:5409])

plt.show()


# # Training and Testing Split

# In[145]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[146]:


from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)


# # Applying SGD Classifier on the entire Dataset

# In[147]:


import gc
from sklearn.metrics import mean_squared_error
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import accuracy_score

estimator = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, 
                          shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                          learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)

print(mean_squared_error(y_test,estimator.fit(X_train,y_train).predict(X_test)))
print ("Accuracy score of model: {}".format(accuracy_score(y_test,estimator.fit(X_train,y_train).predict(X_test))))


# # Reading the Dataset in chunks & applying Partial_fit to check the Training and Testing loss variation

# In[133]:


import numpy as np
chunksize = 4000
estimator = SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, 
                          shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=1, 
                          learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False, 
                          n_iter=10)
trainloss = []
testloss = []
for i,chunk in enumerate(pd.read_csv("cancer2.csv",chunksize=chunksize,header=None,iterator=True)):
    X = chunk.iloc[:,:-1]
    y = chunk.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    estimator.partial_fit(X,y,classes=np.unique(y))
    trainR2 = mean_squared_error(y_train,estimator.predict(X_train))
    testR2 = mean_squared_error(y_test,estimator.predict(X_test))
    trainloss.append(trainR2)
    testloss.append(testR2)
    print("trainloss:{:.4f},testloss:{:.4f} ".format(trainloss[-1],testloss[-1]))
    if i>3:
        break


# In[134]:


import matplotlib.pyplot as plt
plt.plot(trainloss)
plt.plot(testloss)
plt.legend(('train','test'))
plt.show()


# # Reading the Data in chunks and Applying SGDClassifier to check the accuracy of the Model.

# In[238]:


from sklearn.model_selection import train_test_split
y=df['inactive']
X=df[list(df.select_dtypes(include=['float64']))]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[118]:


from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)


# In[233]:


len(X)


# In[239]:


from sklearn.metrics import mean_squared_error
chunksize_load = 5000
chunksize_compute = 10000
from sklearn.metrics import accuracy_score
import pandas as pd
n = len(X)
chunksize = 10000
estimator = SGDClassifier(loss='hinge', penalty='l2',fit_intercept=True)
for epoch in range(5):
    for ii in range(0, n//chunksize_compute):
        X = X_train.iloc[ii*chunksize_compute:(ii+1)*chunksize_compute,:]
        y = y_train.iloc[ii*chunksize_compute:(ii+1)*chunksize_compute]
        estimator.partial_fit(X,y,classes=[0,1])

        print("Accuracy:{}".format(estimator.score(X_test,y_test)))


# # Applying Passive Algorithm Classifier

# In[135]:


from sklearn.linear_model import PassiveAggressiveClassifier
P_estimator = PassiveAggressiveClassifier(C=1.0, fit_intercept=True,shuffle=True, verbose=0, loss='hinge', n_jobs=1, random_state=None, warm_start=False, class_weight=None, n_iter=5)

P_estimator.fit(X_train,y_train)
P_estimator.predict(X_test)
print("Accuracy:{}".format(P_estimator.score(X_test,y_test)))


# # Comparing Different Classifiers

# In[136]:


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
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()


# # Comparing Learning Curves

# In[148]:


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


# In[149]:


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

for name, clf in classifiers:
    title = name
# SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = clf
    plot_learning_curve(clf, title, X_train, y_train, (0.95, 1.01), cv=cv, n_jobs=4)

    plt.show()


# In[ ]:





# In[ ]:





# # PCA as dimensionality reduction

# In[248]:


from sklearn.decomposition import PCA
pca = PCA(n_components=175).fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
print(np.cumsum(pca.explained_variance_ratio_))
plt.show()


# # AS we can see that the variance is about 0.9 if we consider 175 components, so we reduced the dataset dimensionality into 175 features.

# In[ ]:


# from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


pca = PCA(n_components=175)
pca.fit(X)
X_pca = pca.transform(X)

# classifier.fit(X_pca, y)
# pred_labels = classifier.predict(X_pca)


# In[255]:


X_pca.shape


# In[256]:


from sklearn.pipeline import Pipeline

# fits PCA, transforms data and fits the decision tree classifier
# on the transformed data
pipe = Pipeline([('pca', PCA()),
                 ('tree', DecisionTreeClassifier())])

pipe.fit(X_pca, y)

pipe.predict(X_pca)


# In[257]:


from sklearn.cross_validation import cross_val_score
print(cross_val_score(pipe,X_pca, y))


# In[199]:


from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_pca, y, test_size=0.33, random_state=42)


# # Applying SVM on the reduced Data-set

# In[193]:


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



# In[201]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test_1,y_pred))


# In[194]:


title = "Learning Curves (SVM)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = clf_svm
plot_learning_curve(estimator, title, X_train_1, y_train_1, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()


# In[195]:


plot_learning_curve(estimator, title, X_train_1, y_train_1, (0.95, 1.01), cv=cv, n_jobs=4)

plt.show()


# # Applying Decision tree non linear method

# In[204]:


from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state = 0)
clf = clf.fit(X_train_1, y_train_1)
y_pred = clf.predict_proba(X_test_1)


# In[205]:


from sklearn.model_selection import cross_val_score

cross_val_score(clf, X_train_1, y_train_1, cv=10)


# In[279]:


plot_learning_curve(clf, title, X_train_1, y_train_1, (0.95, 1.01), cv=cv, n_jobs=4)

plt.show()


# In[ ]:




