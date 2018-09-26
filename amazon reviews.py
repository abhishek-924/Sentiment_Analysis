
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df=pd.read_csv('C:\ghost drive\study\ds\sentiment analysis\Amazon_Unlocked_Mobile.csv')

#drops empty rows
df.dropna(inplace=True) 


#gives rows where rating is 3
df=df[df['Rating']!=3] 

#create a column where '1' is for rating >3 and '0' is for rating<3
df['positive or negative']=np.where(df['Rating']>3,1,0)


# # training our data
# 

# In[2]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                    df['positive or negative'], 
                                                    random_state=0)
#print(X_train)


 # the bag of words approach
# it do not focuses on structure, but it counts the no. of times the word is repeating

# # CountVectorizer- It allows us to use the bag-of-words approach by converting a collection of text documents into a matrix of token counts. 
# 

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
#it tokenizes
#converts everything to lowercase
#builds a vocabulary
#creates n number of features 
count=CountVectorizer().fit(X_train)
#accessing vocabulary
count.get_feature_names()[::200]
#transforming to matrix representation of bag of words
transform_matrix=count.transform(X_train)
transform_matrix


# In[5]:


#use logistic regression for high dimentional data and for the outcome of only +ve or -ve
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(transform_matrix, y_train)


# In[6]:


from sklearn.metrics import roc_auc_score
predictions=model.predict(count.transform(X_test))
predictions



# In[7]:


print(roc_auc_score(predictions,y_test))


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())


# In[10]:



#X_train_vectorized=vect.transform(X_train)
#LogisticRegression().fit(X_train_vectorized,y_train)
#predictions=LogisticRegression().predict(vect.transform(X_test))


# In[ ]:



                                    


# In[ ]:


#n grams= 1. not an issue, phone is working',2. 'an issue, phone is not working
#both will come -ve by any other method
#n grams solves this situation
#it collects 2,3 words ek saath like bigrams or trigrams and the analysis if it is +ve or -ve
vect = CountVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())


# In[14]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))
predictions



# In[15]:


#ROC (Receiver Operating Characteristic) Curve tells us about how good the 
#model can distinguish between two things (e.g If a patient has a disease or no). 

#for roc_auc score> 90
#ROC curve is large.

print('AUC: ', ((roc_auc_score(y_test, predictions))*100),'%')


# In[32]:


feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()
#sorting bigrams in -ve and +ve way and separating them

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[39]:


print(model.predict(vect.transform(['not an issue, phone is working',
                                    'it is not working fine'])))


# In[ ]:


#it is an example of regression,classification and reinforcement learning 
#it is not an example of clustering

