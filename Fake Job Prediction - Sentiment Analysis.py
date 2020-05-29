
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
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
df = df_initial[['title', 'location', 'company_profile', 'description', 'requirements', 'telecommuting', 'has_company_logo', 'has_questions', 'industry', 'fraudulent']]

# Converting boolean values into String values by replacing 0 with negation String and replacing 1 by Positive String
df['telecommuting'] = df[['telecommuting']].apply(lambda x: 'telecommunicationPresent' if (x['telecommuting'] == 1) else 'telecommunicationAbsent', axis = 1)
df['has_company_logo'] = df[['has_company_logo']].apply(lambda x: 'logoPresent' if (x['has_company_logo'] == 1) else 'logoAbsent', axis = 1)
df['has_questions'] = df[['has_questions']].apply(lambda x: 'questionsPresent' if (x['has_questions'] == 1) else 'questionsAbsent', axis = 1)


# Filling rows having NaN values with empty string
df_new = df.fillna('')

# Combining values of all features in a single column in order to use CountVectorizer and TfidfVectorizer
df_new['combined_features'] = df_new[['title', 'location', 'company_profile', 'description', 'requirements', 'telecommuting', 'has_company_logo', 'has_questions', 'industry']].apply(lambda x: ' '.join(x), axis = 1).str.lower()

# Splliting words into tokens
tokenizer = nltk.RegexpTokenizer(r"\w+")
df_new['combined_features'] = df_new[['combined_features']].apply(lambda x: tokenizer.tokenize(x['combined_features']), axis=1)

# helper function to return stemmed words list from a list of unstemmed words
ps = PorterStemmer()
def get_stemmed_words_list(unstemmed_list):
    str1 = " "
    stemmed_list = []
    for word in unstemmed_list:
        if word.isalpha():
            stemmed_list.append(ps.stem(word))
    return str1.join(stemmed_list)
   
# Stemming words of column 'combined_features'
df_new['combined_features'] = df_new[['combined_features']].apply(lambda x: get_stemmed_words_list(x['combined_features']), axis=1)
   
# Columns to retain
features = ['title', 'location', 'company_profile', 'description', 'requirements', 'industry']
target = ['fraudulent']

# Defining features and target
X = df_new[['combined_features']]
y = df_new['fraudulent']

# Spliting data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 1)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Defining CounterVectorizer
count_vect = CountVectorizer(min_df=3, ngram_range=[1, 2], stop_words='english').fit(X_train['combined_features'])
X_train_count_vect = count_vect.transform(X_train['combined_features'])
X_test_count_vect = count_vect.transform(X_test['combined_features'])

# Gives the number of features
len(count_vect.get_feature_names())



# Defining TfidfVectorizer
tfidf_vect = TfidfVectorizer(min_df=3, ngram_range=[1, 2], stop_words='english').fit(X_train['combined_features'])
X_train_tfidf_vect = tfidf_vect.transform(X_train['combined_features'])
X_test_tfidf_vect = tfidf_vect.transform(X_test['combined_features'])

# Gives the number of features
len(tfidf_vect.get_feature_names())
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
print('\nSmallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

"""
OUTPUT ->

roc_auc_score using LogisticRegression and CounterVectorizer:  0.8681124801658905

Smallest Coefs:
['logopres' 'english' 'logopres questionsabs'
 'telecommunicationabs logopres' 'gr' 'reliabl' 'live' 'sell' 'php'
 'digit']

Largest Coefs: 
['earn' 'logoabs' 'logoabs questionsabs' 'questionsabs account' 'use link'
 'link' 'appli use' 'telecommunicationabs logoabs' 'money' 'ny']

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
print('\nSmallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

"""
OUTPUT ->

roc_auc_score using LogisticRegression and TfidfVectorizer:  0.8624529566427799

Smallest Coefs:
['client' 'team' 'english' 'logopres questionsabs' 'logopres' 'digit'
 'telecommunicationabs logopres' 'grow' 'softwar' 'websit']

Largest Coefs: 
['logoabs' 'telecommunicationabs logoabs' 'logoabs questionsabs' 'earn'
 'use link' 'appli use' 'data entri' 'questionsabs account' 'assist'
 'entri']

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
print('\nSmallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

"""
OUTPUT ->

roc_auc_score using MultinomialNB and CounterVectorizer:  0.9263144930773946

Smallest Coefs:
['limit seo' 'person type' 'person understand' 'person updat' 'person use'
 'person user' 'person valid' 'person vehicl' 'person veri' 'person turn']

Largest Coefs: 
['work' 'experi' 'manag' 'servic' 'skill' 'product' 'requir' 'amp'
 'custom' 'develop']

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
print('\nSmallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

"""
OUTPUT ->

roc_auc_score using MultinomialNB and TfidfVectorizer:  0.8847444908433721

Smallest Coefs:
['limit seo' 'person type' 'person understand' 'person updat' 'person use'
 'person user' 'person valid' 'person vehicl' 'person veri' 'person turn']

Largest Coefs: 
['work' 'logoabs' 'telecommunicationabs logoabs' 'logoabs questionsabs'
 'servic' 'data entri' 'home' 'skill' 'manag' 'custom']

"""

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion
# From the above results based on AURUC score, it can be inferred that the best results are given by Using MultinomialNB and CounterVectorizer together.
