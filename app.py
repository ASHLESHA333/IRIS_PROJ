import numpy as np
import pandas as pd
import pickle
import streamlit as st
##provide header name to browser
st.set_page_config(page_title='Iris Project Ashlesha',layout='wide')
## Add a title in body of browser
st.title('Iris Project')
## take id
id=st.number_input('id:',min_value=0.00,step=0.01)
sep_len=st.number_input('Sepal Length:',min_value=0.00,step=0.01)
sep_wid=st.number_input('Sepal Width:',min_value=0.00,step=0.01)
pet_len=st.number_input('Petal Length:',min_value=0.00,step=0.01)
pet_wid=st.number_input('Petal Width:',min_value=0.00,step=0.01)
submit=st.button('Predict')
st.subheader('Prediction Are: ')
def predict_species(scaler_path,model_path):
    with open(scaler_path,'rb') as file1:
        scaler=pickle.load(file1)
    with open(model_path,'rb') as file2:
        model=pickle.load(file2)
    dct=dct={'id':[id],
        'SepalLengthCm':[sep_len],
         'SepalWidthCm':[sep_wid],
         'PetalLengthCm':[pet_len],
         'PetalWidthCm':[pet_wid]}
    xnew=pd.DataFrame(dct)
    xnew_pre=scaler.transform(xnew)
    pred=model.predict(xnew_pre)
    probs=model.predict_proba(xnew_pre)
    max_prob=np.max(probs)
    return pred,max_prob
## show  the result in streamlit
scaler_path='notebook/scaler.pkl'
model_path='notebook/model.pkl'
pred,max_prob=predict_species(scaler_path,model_path)
st.subheader(f'Prediction Species is:{pred[0]}')
st.subheader(f'Probability of Prediction:{max_prob:.4f}')
st.progress(max_prob)
    