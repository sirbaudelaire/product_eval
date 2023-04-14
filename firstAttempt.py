import sklearn
import numpy as np
import pandas as pd

# Libraries for classical machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("C:/Users/acer/Desktop/python_sht/Sentiment-Analysis-for-Shopee/data/clean_train.csv")
#print(df.head())

X = df['content_stem']
y = df['target']

# Perform train test split so that we can train, score and tune our models' hyperparameters 
#X and y are arrays, stratify is for labels
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Write a function that takes in the actual y value and model predictions, 
# and prints out the confusion matrix and classification report
# Dataset: Validation or test set

def cmat(actual_y, predictions, dataset):
    
    # Create a classification report
    print('Classification report for', dataset)
    print(classification_report(actual_y, predictions))
    print('')
    
    # Create a confusion matrix
    cm = confusion_matrix(actual_y, predictions)
    cm_df = pd.DataFrame(cm, columns=['Predicted Positive Review','Predicted Negative Review'], index=['Actual Positive Review', 'Actual Negative Review'])
    print('Confusion matrix for', dataset)
    print(cm_df)

pipe_tvec_nb = Pipeline([
    ('tvec', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

pipe_tvec_nb.fit(X_train, y_train)
tvec_nb_pred = pipe_tvec_nb.predict(X_val)

# Print accuracy scores
train_score = (pipe_tvec_nb.score(X_train, y_train) * 100) 
val_score = (pipe_tvec_nb.score(X_val, y_val) * 100)
print('Training score:', "{:.2f}".format(train_score))
print('Validation score:', "{:.2f}".format(val_score))
print('')

# Print classification report and confusion matrix
#cmat(y_val, tvec_nb_pred, 'validation set')

# Read test set into a dataframe
test = pd.read_csv("C:/Users/acer/Desktop/python_sht/Sentiment-Analysis-for-Shopee/data/clean_test.csv")
#test = pd.read_csv("C:/Users/acer/Desktop/python_sht/Sentiment-Analysis-for-Shopee/data/shopee_reviews.csv")
#print(test.head())
X_test = test['content_stem']
y_test = test['target']
test_pred = pipe_tvec_nb.predict(X_test)

test_score = (accuracy_score(y_test, test_pred) * 100)
print('Evaluation metrics for test set')
print('')
print('Accuracy score: ', "{:.2f}".format(test_score))
print('')