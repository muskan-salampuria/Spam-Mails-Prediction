#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#Reading Spam Mails File in CSV Froamt
df=pd.read_csv("F:\small_projects\Spam_Mails_Prediction_using_Naive_Bayes\\spam.csv")
df.head()


# In[2]:


#group by the column Category i.e making differnet groups for Ham and Spam mails
df.groupby('Category').describe()


# In[3]:


#introducing a new columns spam whose value is 1 for all spam mails and 0 for all ham mails
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()


# In[4]:


#splitting test and train datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df.Message,df.spam,test_size=0.25)


# In[5]:


#Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
X_train_count=v.fit_transform(X_train.values)
X_train_count.toarray()[:3]


# In[6]:


#Multinomial model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_count,y_train)


# In[7]:


#Testing using an email value
emails={
    'Hey mohan, can we get together to watch football game tommorow?',
    'Upto 20% Discount on parking, exclusive offer just for you. Dont miss this reward!'
}
emails_count=v.transform(emails)
model.predict(emails_count)


# In[9]:


X_test_count=v.transform(X_test)

#calculating Score
model.score(X_test_count,y_test)


# In[10]:


#Pipeline
from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])


# In[11]:


#training clf with the best model betwween Vectorizer and Multinomial NB for this case which is choosen internally.
clf.fit(X_train,y_train)


# In[12]:


#Calculating Score
clf.score(X_test,y_test)

