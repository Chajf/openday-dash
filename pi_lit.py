import streamlit as st
import modele_do as md
import pickle
import pandas as pd
from st_aggrid import AgGrid

st.set_page_config(page_title="Dzień liczby PI", layout="wide")
st.title("Przygotuj własny model uczenia maszynowego")

dane = pd.read_csv('heart.csv', sep=',')

with st.expander("Zobacz dane!"):
    #st.dataframe(dane)
    AgGrid(dane.head(10))


X_train, X_test, y_train, y_test = md.data_prep()


model = st.selectbox("Wybierz który model chcesz stworzyć", ("Regresja logistyczna","Drzewo decyzyjne","Las losowy","XGBoost"))

with st.sidebar:
    st.header("Hiperparametry")
    if model=="Regresja logistyczna":
        C = st.slider("Współczynnik regularyzacji",0.1,1.0)
        pen = st.selectbox("Rodzaj kary", ("l2","Brak"))
    elif model=="Drzewo decyzyjne":
        t1 = st.slider("Max. głębokość",2,10)
        t2 = st.slider("Min. liczba obs. w liściu", 2,10)
    elif model=="Las losowy":
        rf1 = st.slider("Max. głębokość",2,10)
        rf2 = st.selectbox("Kryterium optymalizacji",("gini","entropy"))
    # elif model=="KNN":
    #    mod = md.knn(X_train, y_train,leaf_size=1,n_neighbors=3,p=1)
    elif model=="XGBoost":
        xgb1 = st.slider("Liczba drzew",10,50)
        xgb2 = st.slider("współczynnik nauki", 0.0001, 0.1)
    mod_button = st.button("Naucz model")

if mod_button:
    if model=="Regresja logistyczna":
        if pen=="Brak":
            st.session_state.mod = md.logreg(X_train,y_train,C=C,penalty=None, random_state=42)
        else:
            st.session_state.mod = md.logreg(X_train,y_train,C=C,penalty=pen, random_state=42)
    elif model=="Drzewo decyzyjne":
        st.session_state.mod = md.tree(X_train, y_train, max_depth=t1, min_samples_leaf=t2, random_state=42)
    elif model=="Las losowy":
        st.session_state.mod = md.randfor(X_train, y_train, max_depth=rf1, criterion=rf2, random_state=42)
    # elif model=="KNN":
    #    mod = md.knn(X_train, y_train,leaf_size=1,n_neighbors=3,p=1)
    elif model=="XGBoost":
        st.session_state.mod = md.xgbc(X_train, y_train,n_estimators=xgb1,learning_rate=xgb2)
    #t = tree(X_train, y_train, max_depth=10, min_samples_leaf=10)
        
    with open('mets.pkl', 'rb') as f:
        st.session_state.mets_his = pickle.load(f)
        
    st.session_state.mets = md.dict_metrics(st.session_state.mod, X_test, y_test)
    # with open('mets.pkl', 'rb') as f:
    #     pickle.dump(st.session_state.mets, f)

    with open('mets.pkl', 'wb') as f:
        pickle.dump(st.session_state.mets, f)


if "mod" in st.session_state:

    #pass
    mets = st.session_state.mets
    mets_his = st.session_state.mets_his
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="Acc",value=mets["acc"], delta=f'{mets["acc"]-mets_his["acc"]:.3f}')
    with c3:
        st.metric(label="roc auc",value=mets["roc auc"], delta=f'{mets["roc auc"]-mets_his["roc auc"]:.3f}')

    c1, c2, c3 = st.columns(3)
    with c1:
        #st.metric(label="Acc",value=mets["acc"])
        st.metric(label="precission",value=mets["precission"], delta=f'{mets["precission"]-mets_his["precission"]:.3f}')
    with c2:
        st.metric(label="recall",value=mets["recall"], delta=f'{mets["recall"]-mets_his["recall"]:.3f}')
    with c3:
        #st.metric(label="roc auc",value=mets["roc auc"])
        st.metric(label="f1",value=mets["f1"], delta=f'{mets["f1"]-mets_his["f1"]:.3f}')

    cm = md.produce_confusion(md.conf_mtr(y_test,st.session_state.mod,X_test))
    fpr, tpr = md.m2(st.session_state.mod, X_test, y_test)
    roc = md.produce_roc(fpr,tpr,mets["roc auc"])

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(cm)
    with c2:
        st.altair_chart(roc)

#st.pyplot(md.box1().get_figure(),use_container_width=False)
