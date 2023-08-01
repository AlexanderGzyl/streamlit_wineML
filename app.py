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
alcalinity_of_ash = st.number_input("alcalinity_of_ash", 10,30)
magnesium = st.slider("magnesium  content", 70,165)
total_phenols = st.number_input("total_phenols", 0,1000)
flavanoids = st.slider("flavanoids",0,2)
nonflavanoid_phenols = st.slider("nonflavanoid_phenols", 0.7,6.0, step=0.01)
proanthocyanins = st.number_input("proanthocyanins", 0,1000)
color_intensity = st.number_input("color_intensity", 0,1000)
hue = st.number_input("hue", 0,1000)

proline= st.slider("proline",0,10)
dilution_ratio = st.slider("dilution_ratio ",0,10)

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
