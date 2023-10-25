#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-plot


# In[2]:


pip install wordcloud


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report
from scikitplot.metrics import plot_confusion_matrix


# In[4]:


tsv_file='Restaurant_Reviews.tsv'
 
# reading given tsv file
csv_table=pd.read_table(tsv_file,sep='\t')
 
# converting tsv file into csv
csv_table.to_csv('Restaurant_Reviews.csv',index=False)
 
# output
print("Successfully made csv file")


# In[5]:


csv_table


# In[6]:


df=csv_table


# In[7]:


#Pandas.crosstab():frequency distribution
pd.crosstab(index=df['Liked'],columns='count',dropna=True) #dropna is deleting null values


# In[8]:


#sns.countplot(k.Liked)
sns.countplot(x='Liked', data=df)


# In[9]:


nltk.download('stopwords')
lm = WordNetLemmatizer()
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

corpus = text_transformation(df['Review'])


# In[10]:


df['Review']


# In[11]:


df.isna().sum()


# In[12]:


df.describe()


# In[13]:


plt.rcParams['figure.figsize'] = 20,8
word_cloud = ""
for row in corpus:
    for word in row:
        word_cloud+=" ".join(word)
wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(word_cloud)
plt.imshow(wordcloud)


# In[14]:


cv = CountVectorizer(ngram_range=(1,2))
traindata = cv.fit_transform(corpus)
X = traindata
y = df.Liked


# In[15]:


parameters = {'max_features': ('auto','sqrt'),
             'n_estimators': [500, 1000,1500],
             'max_depth': [5, 10, None],
             'min_samples_split': [5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10],
             'bootstrap': [True, False]}


# In[16]:


grid_search = GridSearchCV(RandomForestClassifier(),parameters,cv=5,return_train_score=True,n_jobs=-1)
grid_search.fit(X,y)
grid_search.best_params_


# In[17]:


'''grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
grid_search.fit(X, y)
print('Best Parameters: ', grid_search.best_params_)

# Check if 'cv_results' is available in the grid_search object
if 'cv_results' in dir(grid_search):
    for i in range(432):
        print('Parameters: ', grid_search.cv_results['params'][i])
        print('Mean Test Score: ', grid_search.cv_results['mean_test_score'][i])
        print('Rank: ', grid_search.cv_results['rank_test_score'][i])
else:
    print("cv_results attribute not found.")'''


# In[18]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Assuming you have defined parameters and X, y before this

# Define the parameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(50, 500, 50),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': np.arange(10, 110, 10),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)

# Fit the model
random_search.fit(X, y)

# Print the best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)


# In[19]:


for i in range(100):
    print('Parameters: ',random_search.cv_results_['params'][i])
    print('Mean Test Score: ',random_search.cv_results_['mean_test_score'][i])
    print('Rank: ',random_search.cv_results_['rank_test_score'][i])


# In[20]:


rfc = RandomForestClassifier(max_features=random_search.best_params_['max_features'],
                                      max_depth=random_search.best_params_['max_depth'],
                                      n_estimators=random_search.best_params_['n_estimators'],
                                      min_samples_split=random_search.best_params_['min_samples_split'],
                                      min_samples_leaf=random_search.best_params_['min_samples_leaf'],
                                      bootstrap=random_search.best_params_['bootstrap'])
rfc.fit(X,y)


# In[21]:


test_df = pd.read_csv('Restaurant_Reviews.csv',delimiter=';',names=['Review','Liked'])
X_test,y_test = test_df.Review,test_df.Liked
#encode the labels into two classes , 0 and 1
#test_df = custom_encoder(y_test)
test_df = df['Liked']
#pre-processing of text
test_corpus = text_transformation(X_test)
#convert text data into vectors
testdata = cv.transform(test_corpus)
#predict the target
predictions = rfc.predict(testdata)


# In[30]:


plt.rcParams['figure.figsize'] = 10,5
#plot_confusion_matrix(y_test,predictions,labels='Liked')
plot=confusion_matrix(y_test,predictions)
acc_score = accuracy_score(y_test,predictions)
pre_score = precision_score(y_test,predictions)
rec_score = recall_score(y_test,predictions)
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print("-"*50)
cr = classification_report(y_test,predictions)
print(cr)


# In[23]:


pip install --user --upgrade scikit-learn


# In[24]:


pip show scikit-learn


# In[25]:


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import plot_confusion_matrix  # Update this line
import matplotlib.pyplot as plt
import numpy as np


# Assuming y_true and predictions are your target variable and predicted labels
# Replace NaN values with a default value or use an appropriate imputation method
y_true = y_true.replace(np.nan, default_value)

# Now you can use the plot_confusion_matrix function
plt.rcParams['figure.figsize'] = 10, 5
plot_confusion_matrix(y_true, predictions)
acc_score = accuracy_score(y_true, predictions)
pre_score = precision_score(y_true, predictions)
rec_score = recall_score(y_true, predictions)
print('Accuracy_score: ', acc_score)
print('Precision_score: ', pre_score)
print('Recall_score: ', rec_score)
print("-" * 50)
cr = classification_report(y_true, predictions)
print(cr)


# In[26]:


predictions_probability = rfc.predict_proba(testdata)
fpr,tpr,thresholds = roc_curve(y_test,predictions_probability[:,1])
plt.plot(fpr,tpr)
plt.plot([0,1])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[ ]:




