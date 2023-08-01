import streamlit as st

import numpy as np
import pandas as pd
import joblib
# get our model
model = joblib.load('extratrees.joblib')

# Title our app
st.title('What region is your Cultivar? :wine_glass:')
# Create  input selectors
alcohol = st.selectbox("Measured Alcohol %",[11,12,13,14,15])
malic_acid = st.slider("Malic Acid %", 0.7,6.0, step=0.01)
ash  = st.slider("ash  %", 1.3,3.3, step=0.01)
alcalinity_of_ash = st.number_input("alcalinity of ash", 10,30)
magnesium = st.slider("magnesium  content", 70,165)
total_phenols = st.slider("total phenols", 0.98,3.88)
flavanoids = st.slider("flavanoids",0.34,5.08)
nonflavanoid_phenols = st.slider("nonflavanoid phenols", 0.13,0.66, step=0.01)
proanthocyanins = st.slider("proanthocyanins", 0.41,3.58)
color_intensity = st.number_input("color intensity", 1,13)
hue = st.slider("hue", 0.48,1.71)
proline= st.number_input("proline",278,1680)
dilution_ratio = st.slider("dilution_ratio ",1.27,4.0)

def predict(): 
    row = np.array([alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins
                    ,color_intensity,hue,dilution_ratio,proline]) 
    X = pd.DataFrame([row], columns = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 
                                        'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 
                                        'od280/od315_of_diluted_wines', 'proline'])
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('From Tuscany :flag-it:')
    elif prediction[0] == 2:  
        st.success('From Abruzzo :flag-it:') 
    else:
        st.success('From Sicily :flag-it:') 
trigger = st.button('Predict', on_click=predict)


# python -m streamlit run app.py 
# https://wineml.streamlit.app/  
