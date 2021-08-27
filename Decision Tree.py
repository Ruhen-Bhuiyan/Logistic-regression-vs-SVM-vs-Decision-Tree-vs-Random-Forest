import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
mydata = pd.read_csv('D:\\445\\Khidmahactuall.csv')
X = mydata.drop('Diagnosis', axis=1)
y = mydata['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y)
depth = np.arange(1, 9)
training_accuracy = np.empty(len(depth))
for i, k in enumerate(depth):
    classi = tree.DecisionTreeClassifier(max_depth=k)
    classi.fit(X_train, y_train)
    training_accuracy[i] = classi.score(X_train, y_train)   
plt.title('Different depth of tree')
plt.plot(depth, training_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
predict = classifier.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predict))




