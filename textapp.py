#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import re
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# In[121]:


url = 'https://raw.githubusercontent.com/libanali12/textapp/main/TheSocialDilemma.csv'
df = pd.read_csv(url)


# In[122]:


user_location = df['user_location'].value_counts().reset_index()
user_location.columns = ['user_location', 'count']
user_location = user_location[user_location['user_location']!='NA']
user_location = user_location.sort_values(['count'],ascending=False)


# In[123]:


user_name = df['user_name'].value_counts().reset_index()
user_name.columns = ['user_name', 'count']
user_name = user_name[user_name['user_name']!='NA']
user_name = user_name.sort_values(['count'],ascending=False)


# In[124]:


source = df['source'].value_counts().reset_index()
source.columns = ['source', 'count']
source = source[source['source']!='NA']
source = source.sort_values(['count'],ascending=False)


# In[125]:


hashtags = df['hashtags'].value_counts().reset_index()
hashtags.columns = ['hashtags', 'count']
hashtags = hashtags[hashtags['hashtags']!='NA']
hashtags = hashtags.sort_values(['count'],ascending=False)


# In[126]:


sentiment = df['Sentiment'].value_counts().reset_index()
sentiment.columns = ['sentiment', 'count']
sentiment = sentiment[sentiment['sentiment']!='NA']
sentiment = sentiment.sort_values(['count'],ascending=False)


# In[133]:


def clean_text(x):
  x = x.lower()
  x = re.sub('\[.*?\]', '', x)
  x = re.sub('https?://\S+|www\.\S+', '', x)
  x = re.sub('\n', '', x)
  x = " ".join(filter(lambda x:x[0]!="@", x.split()))
  return x


# In[134]:


df['text'] = df['text'].apply(lambda x: clean_text(x))


# In[135]:


df['target'] = pd.factorize(df['Sentiment'])[0]


# In[136]:


final_df = df[['text','Sentiment','target']]


# In[137]:


tfidf = TfidfVectorizer(min_df=5,stop_words='english')
scaler = MinMaxScaler()


# In[138]:


features_tfidf = scaler.fit_transform(tfidf.fit_transform(final_df.text).toarray())


# In[139]:


x = features_tfidf
y = final_df['target']


# In[140]:


l_svc1 = LinearSVC(C=1,random_state=0)
nb1 = MultinomialNB(alpha= 1.5, fit_prior= False)
lr1 = LogisticRegression(C= 1.0, penalty= 'l1', solver='liblinear',random_state=0)


# In[141]:


smote = SMOTE()
x_sm,y_sm = smote.fit_resample(x,y)


# In[142]:


accuracies1 = cross_val_score(l_svc1, x_sm, y_sm, scoring='accuracy', cv=3).mean()


# In[143]:


accuracies2 = cross_val_score(nb1, x_sm, y_sm, scoring='accuracy', cv=3).mean()


# In[144]:


accuracies3 = cross_val_score(lr1, x_sm, y_sm, scoring='accuracy', cv=3).mean()


# In[145]:


data = {'Accuracy Score':[accuracies1, accuracies2, accuracies3]}
cv_df = pd.DataFrame(data, index=['LinearSVC', 'MultinomialNB', 'LogisticRegression'])


# In[146]:


lr1.fit(x_sm,y_sm)


# In[147]:


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
        text= st.text_area("Message", height=100)
        if st.button("Predict Tweet Sentiment"):
            result= lr1.predict(tfidf.transform([text]))
            dictionary = {'Neutral':0,'Positive':1,'Negative':2}
            target1 = result.item()
            final_result=[]
            for actual, predicted in dictionary.items():
                if predicted == target1:
                    final_result.append(actual)
            resultdf = pd.DataFrame({'Result':final_result},index=['Sentiment'])
            st.write('Sentiment of your Review')
            st.dataframe(resultdf)
            
if __name__ == '__main__':
    main() 


# In[ ]:




