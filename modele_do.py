# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:13:58 2024

@author: wnadw
"""

import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

def data_prep():
    dane = pd.read_csv('heart.csv', sep=',')
    #dane = dane.drop(columns=["Unnamed: 0"])

    dane.loc[dane['Cholesterol'] == 0, 'Cholesterol'] = pd.NA 
    dane.loc[dane['RestingBP'] == 0, :] = pd.NA

    #one-hot
    dane = pd.get_dummies(dane)

    dane_train, dane_test = train_test_split(dane, test_size = 0.3, random_state = 42) 
    scaler = MinMaxScaler()
    scaler.fit(dane_train)

    dane = pd.DataFrame(scaler.transform(dane), columns = dane.columns)  # scale data
    del dane_train, dane_test, scaler

    #imputer = KNNImputer(n_neighbors=5, weights='distance')
    #dane = pd.DataFrame(imputer.fit_transform(dane), columns=dane.columns)

    dane = dane.dropna()

    #train-test split
    X = dane.loc[:, dane.columns != 'HeartDisease']

    y = dane['HeartDisease']
    y = LabelEncoder().fit_transform(y) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 42)
    return X_train, X_test, y_train, y_test

#budowa modeli

#logistic regression
def logreg(X_train, y_train, C, penalty, random_state = 42):
    lr = LogisticRegression(random_state = random_state, C=C, penalty=penalty)
    lr.fit(X_train, y_train)
    return lr

#decision tree
def tree(X_train, y_train,max_depth,min_samples_leaf, random_state = 42):
    #dt = DecisionTreeClassifier(random_state, max_depth, min_samples_leaf)
    dt = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    dt.fit(X_train, y_train)
    return dt

#random forest
def randfor(X_train, y_train,max_depth = 4,criterion = "gini", random_state = 42):
    rf = RandomForestClassifier(max_depth = max_depth,criterion = criterion)
    rf.fit(X_train, y_train)
    return rf

#kNN
def knn(X_train, y_train, leaf_size = 1, n_neighbors = 3, p = 1):
    knn = KNeighborsClassifier(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    knn.fit(X_train, y_train)
    return knn

#XGBoost
def xgbc(X_train, y_train, n_estimators=100, learning_rate=0.001):
    m_xgb = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    m_xgb.fit(X_train, y_train)
    return m_xgb

#def acc(model, X_test, y_test):
#    y_pred = model.predict(X_test)
#    return accuracy_score(y_true=y_test, y_pred=y_pred)

#wszystkie używane hiperparametry
hyperparameters = {
    "n_estimators"
    "learning_rate"
    "p"
    "n_neighbors"
    "leaf_size"
    "max_depth"
    "criterion"
    "random_state"
    "min_samples_leaf"
    "penalty"
    "C"
}


#metryki
def dict_metrics(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    metr = {
        "acc": round(acc,3),
        "roc auc": round(roc_auc,3),
        "precission": round(precision,3),
        "recall": round(recall,3),
        "f1": round(f1,3)
        }
    return metr



#confiusion matrix
def conf_mtr(y_test, model, X_test):
    cm = confusion_matrix(y_true=y_test, y_pred=model.predict(X_test))
    return cm

def produce_confusion(cm):
    """Given the confusion matrix array output from SKLearn, create an Altair heatmap"""

    data = pd.DataFrame(
        {
            "Actual": np.array(["Positive", "Negative", "Positive", "Negative"]),
            "Predicted": np.array(["Positive", "Negative", "Negative", "Positive"]),
            "Count": np.array([cm[0, 0], cm[1, 1], cm[1, 0], cm[0, 1]]),
            "Color": np.array(
                ["#66BB6A", "#66BB6A", "#EF5350", "#EF5350"]
            ),  # Customize the hex colors here
        }
    )

    # Create a heatmap with appropriate colors
    heatmap = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x="Actual:N",
            y="Predicted:N",
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Actual:N", "Predicted:N", "Count:Q"],
        )
        .properties(title="Confusion Matrix", width=420, height=520)
    )

    # Add text to display the count
    text = (
        alt.Chart(data)
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x="Actual:N",
            y="Predicted:N",
            text="Count:Q",
            color=alt.condition(
                alt.datum.Color == "#EF5350", alt.value("white"), alt.value("black")
            ),
        )
    )

    # Combine heatmap and text layers
    chart = (heatmap + text).configure_title(
        fontSize=18, fontWeight="bold", anchor="middle"
    )

    return chart

def m2(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fpr,tpr,_ = roc_curve(y_test, y_scores)
    return fpr, tpr

#roc
def produce_roc(fpr, tpr, roc_auc):
    roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
    roc_chart = (
        alt.Chart(roc_data)
        .mark_line()
        .encode(
            x=alt.X("False Positive Rate", title="False Positive Rate (FPR)"),
            y=alt.Y("True Positive Rate", title="True Positive Rate (TPR)"),
        )
        .properties(title=f"ROC Curve (AUC = {roc_auc:.2f})",width=520, height=520)
    )
    return roc_chart






#kolumny do wykorzystania
#Age - wiek (numeric)
#Sex - płeć (categorical)
#ChestPainType - typ bólu klatki piersiowej (categorical)
#RestingBP - spoczynkowe ciśnienie krwi (numeric)
#Cholesterol - cholesterol w surowicy (numeric)
#FastingBS - poziom cukru we krwi naczczo (categorical)
#RestingECG - wyniki elektrokardiogramu spoczynkowego (categorical)
#MaxHR - osiągnięte tętno maksymalne (numeric)
#ExerciseAngina - categorical 
#Oldpeak - wskaźnik ST (mierzony w depresji) (numeric)
#ST_Slope - nachylenie szczytowego wysiłkowego odcinka ST (categorical)
#HeartDisease - ma lub nie ma chorobę serca (categorical)

def box1():
    dane = pd.read_csv('heart.csv', sep=',')
    dane.loc[dane['Cholesterol'] == 0, 'Cholesterol'] = pd.NA 
    dane.loc[dane['RestingBP'] == 0, :] = pd.NA

    sns.set(rc={'figure.figsize':(5,5)})
    sns.set_style("white")
    #sns.boxplot(x='HeartDisease', y='Age', data=dane)
    # sns.boxplot(x='Sex', y='Cholesterol', data=dane)
    # sns.boxplot(x='ChestPainType', y='Age', data=dane)
    # sns.boxplot(x='HeartDisease', y='MaxHR', data=dane)
    # sns.boxplot(x='HeartDisease', y='RestingBP', data=dane)
    # sns.boxplot(x='FastingBS', y='Age', data=dane)
    # sns.boxplot(x='Sex', y='MaxHR', data=dane)
    # sns.boxplot(x='Sex', y='Age', data=dane)
    ax = sns.boxplot(x="Age", y="ChestPainType", data=dane, orient='h')
    return ax

def test_f():
    dane = pd.read_csv('heart.csv', sep=',', index_col=0)
    dane.loc[dane['Cholesterol'] == 0, 'Cholesterol'] = pd.NA 
    dane.loc[dane['RestingBP'] == 0, :] = pd.NA

    #wykresy (z różnymi kombinacjami zmiennych - do wyboru)

    # boxplot
    sns.set(rc={'figure.figsize':(10,10)})
    sns.set_style("white")
    sns.boxplot(x='HeartDisease', y='Age', data=dane)
    sns.boxplot(x='Sex', y='Cholesterol', data=dane)
    sns.boxplot(x='ChestPainType', y='Age', data=dane)
    sns.boxplot(x='HeartDisease', y='MaxHR', data=dane)
    sns.boxplot(x='HeartDisease', y='RestingBP', data=dane)
    sns.boxplot(x='FastingBS', y='Age', data=dane)
    sns.boxplot(x='Sex', y='MaxHR', data=dane)
    sns.boxplot(x='Sex', y='Age', data=dane)
    ax = sns.boxplot(x="Age", y="ChestPainType", data=dane, orient='h')

    # scatter plot
    plt.scatter(dane['Age'], dane['Cholesterol'])
    plt.scatter(dane['Cholesterol'], dane['RestingBP'])
    plt.scatter(dane['Age'], dane['MaxHR'])

    #3D plot
    # assign axis values
    x = dane["Age"]
    y = dane["Cholesterol"]
    z = dane["RestingBP"]
    # adjust size of plot
    sns.set(rc={'figure.figsize': (8, 5)})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    # assign labels
    ax.set_xlabel('Age'), ax.set_ylabel('Cholesterol'), ax.set_zlabel('RestingBP')
    # display illustration
    plt.show()


    #histogram
    features = ['Age', 'Cholesterol']
    dane[features].hist(figsize=(10, 4))
    features = ['Age', 'MaxHR']
    dane[features].hist(figsize=(10, 4))
    features = ['Age', 'RestingBP']
    dane[features].hist(figsize=(10, 4))
    features = ['RestingBP', 'Cholesterol']
    dane[features].hist(figsize=(10, 4))
    features = ['RestingBP', 'MaxHR']
    dane[features].hist(figsize=(10, 4))
    features = ['MaxHR', 'Cholesterol']
    dane[features].hist(figsize=(10, 4))


    #countplots
    #dwa oddzielnie
    sns.countplot(x='HeartDisease', data=dane)
    sns.set(rc={'figure.figsize':(26,10)})
    sns.countplot(x='Age', data=dane)
    #lub na jednym plocie:
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(26, 6)) #możnaby pozmieniać szerokość aby dobrze było widać zakres 'Age'
    sns.countplot(x='HeartDisease', data=dane, ax=axes[0])
    sns.countplot(x='Age', data=dane, ax=axes[1])
    #inne countplots
    sns.set(rc={'figure.figsize':(10,5)})
    sns.countplot(x='Sex', data=dane)
    sns.countplot(x='FastingBS', data=dane)

    #heatmap
    sns.set(rc={'figure.figsize':(10,6)})
    sns.heatmap(dane.corr(),
                cmap=sns.cubehelix_palette(20, light=0.95, dark=0.15))


