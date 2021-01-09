import numpy as np
import pandas as pd
import nltk
df=pd.read_csv('spam.csv',encoding='latin')
print(df.head())
print(df.shape)

df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
print(df.head())

df_copy=df['v2'].copy()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
vectorizer.fit(df_copy)
vector=vectorizer.transform(df_copy)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
v2_train,v2_test,v1_train,v1_test=train_test_split(vector,df['v1'],test_size=0.15,random_state=)
Spam_model=LogisticRegression(solver='liblinear')
Spam_model=
