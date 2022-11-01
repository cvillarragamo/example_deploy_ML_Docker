import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from joblib import dump, load

st.set_page_config(page_title = 'Streamlit',layout = 'wide')

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

st.markdown("<h1 style='text-align: center; color:black;'> ML model deployment </h1>",unsafe_allow_html=True)
st.markdown("<br></br>",unsafe_allow_html=True)
              
image = Image.open('image/flores.jpg')
st.image(image)


with st.sidebar:
    clasificador = st.selectbox(label='Clasificador',options=('Decision Tree Classifier','K-Nearest Neighbors'))
    st.markdown("<br></br>",unsafe_allow_html=True)

    sepal_length = st.slider(label='Sepal length (cm)',min_value=2.0,max_value=9.0, step=0.05)
    sepal_width = st.slider(label='Sepal width (cm)',min_value=1.0,max_value=6.0,step=0.05)
    petal_length = st.slider(label='Petal length (cm)',min_value=0.0,max_value=8.0,step=0.05)
    petal_width = st.slider(label='Petal width (cm)',min_value=0.0,max_value=8.0,step=0.05)

def classifier(clasificador):
    if clasificador=='Decision Tree Classifier':
        tree_classifier = load('model/tree_model.py')
        resultado = tree_classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
        
    elif clasificador=='K-Nearest Neighbors':
        kneighbors_classifier = load('model/kneighbors_model.py')
        resultado = kneighbors_classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    else:
        resultado = None
    return resultado

col1,col2,col3 = st.columns(3)

flecha = Image.open('image/flecha.png')

if st.button('¡Clasifica!'):
    clase = int(classifier(clasificador))
    if clase==0:
        col1.image(flecha)
    elif clase==1:
        col2.image(flecha)
    elif clase==2:
        col3.image(flecha)
    else:
        pass
else:
    st.write('Elija clasificador y parámetros')






