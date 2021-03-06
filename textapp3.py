#!/usr/bin/env python
# coding: utf-8

# In[475]:


import pandas as pd 
import numpy as np
import streamlit as st
import plotly.express as px
import re
from io import BytesIO
import requests
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# In[476]:


@st.experimental_memo
def dataframe():
    url = 'https://raw.githubusercontent.com/libanali12/textapp/main/TheSocialDilemma.csv'    
    data = pd.read_csv(url)
    return data


# In[477]:


df=dataframe()


# In[478]:


user_location = df['user_location'].value_counts().reset_index()
user_location.columns = ['user_location', 'count']
user_location = user_location[user_location['user_location']!='NA']
user_location = user_location.sort_values(['count'],ascending=False)


# In[479]:


user_name = df['user_name'].value_counts().reset_index()
user_name.columns = ['user_name', 'count']
user_name = user_name[user_name['user_name']!='NA']
user_name = user_name.sort_values(['count'],ascending=False)


# In[480]:


source = df['source'].value_counts().reset_index()
source.columns = ['source', 'count']
source = source[source['source']!='NA']
source = source.sort_values(['count'],ascending=False)


# In[481]:


hashtags = df['hashtags'].value_counts().reset_index()
hashtags.columns = ['hashtags', 'count']
hashtags = hashtags[hashtags['hashtags']!='NA']
hashtags = hashtags.sort_values(['count'],ascending=False)


# In[482]:


sentiment = df['Sentiment'].value_counts().reset_index()
sentiment.columns = ['sentiment', 'count']
sentiment = sentiment[sentiment['sentiment']!='NA']
sentiment = sentiment.sort_values(['count'],ascending=False)


# In[483]:


@st.experimental_singleton
def clean_text(x):
  x = x.lower()
  x = re.sub('\[.*?\]', '', x)
  x = re.sub('https?://\S+|www\.\S+', '', x)
  x = re.sub('\n', '', x)
  x = " ".join(filter(lambda x:x[0]!="@", x.split()))
  return x


# In[484]:


@st.experimental_singleton
def text1():
    df['text'] = df['text'].apply(lambda x: clean_text(x))
    text2 = df['text']
    return text2


# In[485]:


df['target'] = pd.factorize(df['Sentiment'])[0]


# In[486]:


textdf = text1()


# In[501]:


tfidf = TfidfVectorizer(min_df=5,stop_words='english')
scaler = MinMaxScaler()
features_tfidf = scaler.fit_transform(tfidf.fit_transform(textdf).toarray())


# In[502]:


x = features_tfidf
y = df['target']


# In[503]:


@st.experimental_singleton
def features(x,y):
    oversampler = RandomOverSampler()
    x_sm,y_sm = oversampler.fit_resample(x,y)
    return x_sm,y_sm


# In[504]:


@st.experimental_singleton
def load_model():
    link = "https://raw.githubusercontent.com/libanali12/textapp/main/model.pkl"
    mfile = BytesIO(requests.get(link).content)
    model= pickle.load(mfile)
    return model


# In[505]:


x_sm,y_sm = features(x,y)
lr1 = load_model()


# In[508]:


def main():
    st.title("Social Dilemma Tweet Classification App")
    menu = ["Home","Model"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        st.write('Social Dilemma Tweet Dataset')
        st.dataframe(df.head(5))
        st.subheader("Exploratory Data Analysis")
        fig1 = px.bar(user_location,x=user_location.head(10)["count"], y=user_location.head(10)["user_location"],
             color_discrete_sequence=px.colors.qualitative.Alphabet,
             height=600, width=900)
        fig1.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig1.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig1.update_layout(showlegend=False, title="Top 10 user locations",
                  xaxis_title="Count",
                  yaxis_title="user_location")
        fig1.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig1.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig1)
        fig2 = px.bar(user_name,x=user_name.head(10)["count"], y=user_name.head(10)["user_name"],
             color_discrete_sequence=px.colors.qualitative.Alphabet,
             height=600, width=900)
        fig2.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig2.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig2.update_layout(showlegend=False, title="Top 10 user based on number of tweets",
                  xaxis_title="Count",
                  yaxis_title="user_name")
        fig2.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig2.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig2)
        fig3 = px.bar(source,x=source.head(10)["count"], y=source.head(10)["source"],
             color_discrete_sequence=px.colors.qualitative.Alphabet,
             height=600, width=900)
        fig3.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig3.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig3.update_layout(showlegend=False, title="Top 10 device used to make tweets",
                  xaxis_title="Count",
                  yaxis_title="source")
        fig3.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig3.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig3)
        fig4 = px.bar(hashtags,x=hashtags.head(5)["count"], y=hashtags.head(5)["hashtags"],
             color_discrete_sequence=px.colors.qualitative.Alphabet,
             height=600, width=900)
        fig4.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig4.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig4.update_layout(showlegend=False, title="Top 5 hashtags",
                  xaxis_title="Count",
                  yaxis_title="hashtags")
        fig4.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig4.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig4)
        fig5 = px.bar(sentiment,x=sentiment.head(3)["count"], y=sentiment.head(3)["sentiment"],
             color_discrete_sequence=px.colors.qualitative.Alphabet,
             height=600, width=900)
        fig5.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig5.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig5.update_layout(showlegend=False, title="Sentiment Breakdown",
                  xaxis_title="Count",
                  yaxis_title="Sentiment")
        fig5.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig5.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig5)
    else:
        st.subheader("Social Dilemma Tweet Classification Model")
        text= st.text_area("Write your review below", height=100)
        if st.button("Predict Tweet Sentiment"):
            result= lr1.predict(tfidf.transform([text]))
            resultdf = pd.DataFrame({'Result':result},index=['Sentiment'])
            resultdf['Class'] = ['Neutral' if x == 0 else 'Postive' if x == 1 else 'Negative' for x in resultdf['Result']]
            st.write('Sentiment of your Review')
            st.dataframe(resultdf)
            
if __name__ == '__main__':
    main() 


# In[ ]:




