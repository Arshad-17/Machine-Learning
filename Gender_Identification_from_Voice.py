#Importing Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

#Importing the dataset
data = pd.read_csv('voice.csv')
data.head()

#Checking for missing values in the dataset
data.isnull().sum()
data.describe()

#Encoding male:1 and female:0
data = pd.get_dummies(data,prefix=None,drop_first=True)

#Visualizing the features in the dataset
import seaborn as sns
import matplotlib.pyplot as plt
plt.subplots(4,5,figsize=(15,15))
for i in range(1,21):
   plt.subplot(4,5,i)
   plt.title(data.columns[i-1])
   sns.kdeplot(data.loc[data['label_male'] == 0 , data.columns[i-1]], color= 'green', label='F')
   sns.kdeplot(data.loc[data['label_male'] == 1 , data.columns[i-1]], color= 'blue', label='M')



#Splitting dataset into training and test sets
from sklearn.model_selection import train_test_split
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
X_train, X_test, y_train, y_test =    train_test_split(X, y,  test_size=0.3, 
                                                             random_state=0, 
                                                                 stratify=y)
print(y_train.shape,X_train.shape,y_test.shape,X_test.shape

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train_std, y_train)
print("Logistic Regression")
print("Accuracy on training set: {:.4f}".format(logreg.score(X_train_std, y_train)))
print("Accuracy on test set: {:.4f}".format(logreg.score(X_test_std, y_test)))
y_pred = logreg.predict(X_test_std)

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print("Precision, Recall and fscore:",precision, recall, fscore,)
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True,cmap='Blues',fmt='')


print("Coeffiient:",logreg.coef_)
print("Intercept:",logreg.intercept_)

#Train decision tree model

DTclf = DecisionTreeClassifier(criterion='gini',random_state=0,max_depth=4)
DTclf.fit(X_train_std, y_train)

print("Decision Tree")
print("Accuracy on training set: {:.4f}".format(DTclf.score(X_train_std, y_train)))
print("Accuracy on test set: {:.4f}".format(DTclf.score(X_test_std, y_test)))

y_pred = DTclf.predict(X_test_std)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print("Precision, Recall and fscore:",precision, recall, fscore,)
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True,cmap='Blues',fmt='')


a=['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
       'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
       'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']
b=['male','female']


#Visualizing the decision tree
fig = plt.figure(figsize=(20,20))
_ = tree.plot_tree(DTclf, feature_names= a,class_names= b,filled=True)
fig.savefig("decistion_tree.png")


#Train random forest model

forest = RandomForestClassifier(n_estimators=5, random_state=0)
forest.fit(X_train_std, y_train)

print("Random Forest")
print("Accuracy on training set: {:.4f}".format(forest.score(X_train_std, y_train)))
print("Accuracy on test set: {:.4f}".format(forest.score(X_test_std, y_test)))

y_pred = forest.predict(X_test_std)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print("Precision, Recall and fscore:",precision, recall, fscore,)
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True,cmap='Blues',fmt='')


forest.base_estimator_

#Train support vector machine model

svm = SVC(kernel='rbf')
svm.fit(X_train_std, y_train)

print("Support Vector Machine")
print("Accuracy on training set: {:.4f}".format(svm.score(X_train_std, y_train)))
print("Accuracy on test set: {:.4f}".format(svm.score(X_test_std, y_test)))

y_pred_sm = svm.predict(X_test_std)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print("Precision, Recall and fscore:",precision, recall, fscore,)
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True,cmap='Blues',fmt='')


svm.support_vectors_

#Plot the graph for feature selection for decision tree and random forest
def graph_feature_importances(model):
    n_features = X_train_std.shape[1]
    plt.figure(figsize=(8,6))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), list(data))
    plt.title("Feature Selection")
    plt.xlabel("Variable importance")
    plt.ylabel("Independent Variable")
    plt.show()

graph_feature_importances(DTclf)
graph_feature_importances(forest)

