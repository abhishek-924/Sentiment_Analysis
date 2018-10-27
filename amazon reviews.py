import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('C:\ghost drive\study\ds\sentiment analysis\Amazon_Unlocked_Mobile.csv')
df.dropna(inplace=True) 
df=df[df['Rating']!=3] 
df['positive or negative']=np.where(df['Rating']>3,1,0)
#random experiment
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['positive or negative'], random_state=0)
print(X_train)


from sklearn.feature_extraction.text import CountVectorizer

count=CountVectorizer().fit(X_train)
count.get_feature_names()[::200]

transform_matrix=count.transform(X_train)
transform_matrix

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(transform_matrix, y_train)

from sklearn.metrics import roc_auc_score
predictions=model.predict(count.transform(X_test))
predictions

print(roc_auc_score(predictions,y_test))
#imma get that tshirt

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())

vect = CountVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
len(vect.get_feature_names())

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))
predictions

print('AUC: ', ((roc_auc_score(y_test, predictions))*100),'%')

feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'it is not working fine'])))

