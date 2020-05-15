
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score


# Reading Data in the dataframe
df_initial = pd.read_csv(r'...data/fake_job_postings.csv')
# df_initial.head()

# Columns in dataset df_initial
# df_initial.columns

# Creating df with selected/important columns
df = df_initial[['title', 'location', 'company_profile', 'description', 'requirements', 'industry', 'fraudulent']]
# Dropping rows having NaN values
df_new = df.dropna()

# Columns to retain
features = ['title', 'location', 'company_profile', 'description', 'requirements', 'industry']
target = ['fraudulent']

X = df_new[features]
y = df_new['fraudulent']

# Spliting data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 1)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Defining CounterVectorizer
count_vect = CountVectorizer(min_df=3, ngram_range=[1, 2]).fit(X_train['title']+' '+X_train['location']+' '+X_train['company_profile']+' '+X_train['description']+' '+X_train['requirements']+' '+X_train['industry'])
X_train_count_vect = count_vect.transform(X_train['title']+' '+X_train['location']+' '+X_train['company_profile']+' '+X_train['description']+' '+X_train['requirements']+' '+X_train['industry'])
X_test_count_vect = count_vect.transform(X_test['title']+' '+X_test['location']+' '+X_test['company_profile']+' '+X_test['description']+' '+X_test['requirements']+' '+X_test['industry'])



# Defining TfidfVectorizer
tfidf_vect = TfidfVectorizer(min_df=3, ngram_range=[1, 2]).fit(X_train['title']+' '+X_train['location']+' '+X_train['company_profile']+' '+X_train['description']+' '+X_train['requirements']+' '+X_train['industry'])
X_train_tfidf_vect = tfidf_vect.transform(X_train['title']+' '+X_train['location']+' '+X_train['company_profile']+' '+X_train['description']+' '+X_train['requirements']+' '+X_train['industry'])
X_test_tfidf_vect = tfidf_vect.transform(X_test['title']+' '+X_test['location']+' '+X_test['company_profile']+' '+X_test['description']+' '+X_test['requirements']+' '+X_test['industry'])

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Using LogisticRegression and CounterVectorizer for calculating roc_auc_score
lr_cv_model = LogisticRegression(C=0.1, max_iter=500, random_state = 1).fit(X_train_count_vect, y_train)
y_pred_lr_cv = lr_cv_model.predict(X_test_count_vect)
score_lr_cv = roc_auc_score(y_test, y_pred_lr_cv)
print("roc_auc_score using LogisticRegression and CounterVectorizer: ", score_lr_cv)

# get the feature names as numpy array
feature_names = np.array(count_vect.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index = lr_cv_model.coef_[0].argsort()
# Find the 10 smallest and 10 largest coefficients
print('\n\nSmallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

"""
OUTPUT ->

roc_auc_score using LogisticRegression and CounterVectorizer:  0.96


Smallest Coefs:
['based' 'it' 'is' 'we' 'recruitment' 'an' 'staff' 'you' 'are' 'english']

Largest Coefs: 
['financing' 'accion' 'hotel' 'engineering' 'industry' 'controls' 'is to'
 'more' 'tx' 'candidates']

"""

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Using LogisticRegression and TfidfVectorizer for calculating roc_auc_score
lr_tfidf_model = LogisticRegression(C=15, max_iter=500, random_state = 1).fit(X_train_tfidf_vect, y_train)
y_pred_lr_tfidf = lr_tfidf_model.predict(X_test_tfidf_vect)
score_lr_tfidf = roc_auc_score(y_test, y_pred_lr_tfidf)
print("roc_auc_score using LogisticRegression and TfidfVectorizer: ", score_lr_tfidf)

# get the feature names as numpy array
feature_names = np.array(tfidf_vect.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index = lr_tfidf_model.coef_[0].argsort()
# Find the 10 smallest and 10 largest coefficients
print('\n\nSmallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

"""
OUTPUT ->

roc_auc_score using LogisticRegression and TfidfVectorizer:  0.94


Smallest Coefs:
['you' 'we' 'it' 'of' 'is' 'social' 'are' 'be' 're' 'based']

Largest Coefs: 
['accion' 'hotel' 'engineering' 'aptitude staffing' 'american secured'
 'novation' 'financing' 'controls' 'oil' 'your career']

"""

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Using MultinomialNB and CounterVectorizer for calculating roc_auc_score
mnb_cv_model = MultinomialNB(alpha=0.1).fit(X_train_count_vect, y_train)
y_pred_mnb_cv = mnb_cv_model.predict(X_test_count_vect)
score_mnb_cv = roc_auc_score(y_test, y_pred_mnb_cv)
print("roc_auc_score using MultinomialNB and CounterVectorizer: ", score_mnb_cv)

# get the feature names as numpy array
feature_names = np.array(count_vect.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index = mnb_cv_model.coef_[0].argsort()
# Find the 10 smallest and 10 largest coefficients
print('\n\nSmallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

"""
OUTPUT ->

roc_auc_score using MultinomialNB and CounterVectorizer:  0.97


Smallest Coefs:
['like code' 'perform successfully' 'perform tasks' 'perform testing'
 'perform their' 'perform these' 'perform this' 'perform start'
 'perform thorough' 'perform under']

Largest Coefs: 
['and' 'to' 'the' 'of' 'in' 'with' 'for' 'our' 'is' 'experience']

"""

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Using MultinomialNB and TfidfVectorizer for calculating roc_auc_score
mnb_tfidf_model = MultinomialNB(alpha=0.01).fit(X_train_tfidf_vect, y_train)
y_pred_mnb_tfidf = mnb_tfidf_model.predict(X_test_tfidf_vect)
score_mnb_tfidf = roc_auc_score(y_test, y_pred_mnb_tfidf)
print("roc_auc_score using MultinomialNB and TfidfVectorizer: ", score_mnb_tfidf)


# get the feature names as numpy array
feature_names = np.array(tfidf_vect.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index = mnb_tfidf_model.coef_[0].argsort()
# Find the 10 smallest and 10 largest coefficients
print('\n\nSmallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

"""
OUTPUT ->

roc_auc_score using MultinomialNB and TfidfVectorizer:  0.97


Smallest Coefs:
['like code' 'perform successfully' 'perform tasks' 'perform testing'
 'perform their' 'perform these' 'perform this' 'perform start'
 'perform thorough' 'perform under']

Largest Coefs: 
['and' 'to' 'the' 'of' 'in' 'with' 'for' 'oil' 'our' 'engineering']

"""

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
