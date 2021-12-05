#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression


# In[2]:


st.title('Model Deployment: Bankruptcy Prevention')


# In[3]:


st.sidebar.header('User Input Parameters')


# In[4]:


def user_input_features():
    industrial_risk = st.sidebar.selectbox('industrial_risk',('1','0','0.5'))
    management_risk = st.sidebar.selectbox('management_risk',('1','0','0.5'))
    financial_flexibility = st.sidebar.selectbox('financial_flexibility',('1','0','0.5'))
    credibility = st.sidebar.selectbox('credibility',('1','0','0.5'))
    competitiveness = st.sidebar.selectbox('competitiveness',('1','0','0.5'))
    operating_risk = st.sidebar.selectbox('operating_risk',('1','0','0.5'))
    data = {'industrial_risk':industrial_risk,
            'management_risk':management_risk,
            'financial_flexibility':financial_flexibility,
            'credibility':credibility,
            'competitiveness':competitiveness,
            'operating_risk':operating_risk}
    features = pd.DataFrame(data,index = [0])
    return features


# In[5]:


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# In[6]:


data = pd.read_csv("E:/Data science/DS project/bank.csv")
data['class'] = data['class'].astype("category")
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])


# In[7]:


data['class']


# In[8]:


X = data.iloc[:,0:6]
Y = data.iloc[:,6]
clf = LogisticRegression()
clf.fit(X,Y)


# In[9]:


prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('non-bankcruptcy' if prediction_proba[0][1] > 0.5 else 'bankruptcy')

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:





# In[ ]:




