import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble  import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn import preprocessing

results = open("results.txt", "w")

data = pd.read_csv("bank-numbers.csv", sep=",")
x = data.iloc[:, :16]
y = data.iloc[:, 16]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

x_train_scaled = preprocessing.scale(x_train)
x_test_scaled = preprocessing.scale(x_test)

perceptron = Perceptron()
perceptron.fit(x_train_scaled, y_train)
pred_perceptron = perceptron.predict(x_test_scaled)
confusion_perceptron = confusion_matrix(y_test, pred_perceptron)
results.write("Confusion matrix - perceptron: \n{}\n".format(confusion_perceptron))
results.write("f1-metrics of perceptron: {:.2f}\n".format(f1_score(y_test, pred_perceptron)))
results.write("Perceptron classification report:\n")
results.write(classification_report(y_test, pred_perceptron, target_names=["yes", "no"]))
results.write("Perceptron score: {:.2f}\n\n".format(perceptron.score(x_test_scaled, y_test)))


lg = LogisticRegression()
lg.fit(x_train_scaled, y_train)
pred_lg = lg.predict(x_test_scaled)
confusion_lg = confusion_matrix(y_test, pred_lg)
results.write("Confusion matrix - logistic regression: \n{}\n".format(confusion_lg))
results.write("f1-metrics of logistic regression: {:.2f}\n".format(f1_score(y_test, pred_lg)))
results.write("Logistic regression classification report:\n")
results.write(classification_report(y_test, pred_lg, target_names=["yes", "no"]))
results.write("Logistic regression score: {:.2f}\n\n".format(lg.score(x_test_scaled, y_test)))


knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)
pred_knn = knn.predict(x_test_scaled)
confusion_knn = confusion_matrix(y_test, pred_knn)
results.write("Confusion matrix - KNN: \n{}\n".format(confusion_knn))
results.write("f1-metrics of KNN: {:.2f}\n".format(f1_score(y_test, pred_knn)))
results.write("KNN classification report:\n")
results.write(classification_report(y_test, pred_knn, target_names=["yes", "no"]))
results.write("KNN score: {:.2f}\n\n".format(knn.score(x_test_scaled, y_test)))


decisionTree = DecisionTreeClassifier()
decisionTree.fit(x_train_scaled, y_train)
pred_decisionTree = decisionTree.predict(x_test_scaled)
confusion_decisionTree = confusion_matrix(y_test, pred_decisionTree)
results.write("Confusion matrix - decision tree: \n{}\n".format(confusion_decisionTree))
results.write("f1-metrics of decision tree: {:.2f}\n".format(f1_score(y_test, pred_decisionTree)))
results.write("Decision tree classification report:\n")
results.write(classification_report(y_test, pred_decisionTree, target_names=["yes", "no"]))
results.write("Desicion tree score: {:.2f}\n\n".format(decisionTree.score(x_test_scaled, y_test)))


naive = BernoulliNB()
naive.fit(x_train_scaled, y_train)
pred_naive = naive.predict(x_test_scaled)
confusion_naive = confusion_matrix(y_test, pred_naive)
results.write("Confusion matrix - naive Bayes: \n{}\n".format(confusion_naive))
results.write("f1-metrics of naive Bayes: {:.2f}\n".format(f1_score(y_test, pred_naive)))
results.write("Naive Bayes classification report: \n")
results.write(classification_report(y_test, pred_naive, target_names=["yes", "no"]))
results.write("Naive Bayes score: {:.2f}\n\n".format(naive.score(x_test_scaled, y_test)))


rfc = RandomForestClassifier()
rfc.fit(x_train_scaled, y_train)
pred_rfc = rfc.predict(x_test_scaled)
confusion_rfc = confusion_matrix(y_test, pred_rfc)
results.write("Confusion matrix - RFC: \n{}\n".format(confusion_rfc))
results.write("f1-metrics of RFC: {:.2f}\n".format(f1_score(y_test, pred_rfc)))
results.write("RFC classification report:\n")
results.write(classification_report(y_test, pred_rfc, target_names=["yes", "no"]))
results.write("Random forest score: {:.2f}\n\n".format(rfc.score(x_test_scaled, y_test)))


svc = SVC()
svc.fit(x_train_scaled, y_train)
pred_svc = svc.predict(x_test_scaled)
confusion_svc = confusion_matrix(y_test, pred_svc)
results.write("Confusion matrix - SVC: \n{}\n".format(confusion_svc))
results.write("f1-metrics of SVC: {:.2f}\n".format(f1_score(y_test, pred_svc)))
results.write("SVC classification report:\n")
results.write(classification_report(y_test, pred_svc, target_names=["yes", "no"]))
results.write("Support vector score: {:.2f}\n\n".format(svc.score(x_test_scaled, y_test)))

results.close()
print("END")
