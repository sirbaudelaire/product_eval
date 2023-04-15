import sklearn
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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

def predict(raw_text: str):

    # Instantiate PorterStemmer
    p_stemmer = PorterStemmer()

    # Remove HTML
    #review_text = BeautifulSoup(raw_text, features="lxml").get_text()

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)

    # Convert words to lower case and split each word up
    words = letters_only.lower().split()

    # Convert stopwords to a set
    stops = set(stopwords.words('english'))

    # Adding on stopwords that were appearing frequently in both positive and negative reviews
    stops.update(['app','shopee','shoppee','item','items','seller','sellers','bad'])

    # Remove stopwords
    meaningful_words = [w for w in words if w not in stops]

    # Stem words
    meaningful_words = [p_stemmer.stem(w) for w in meaningful_words]

    # Join words back into one string, with a space in between each word
    final_text = pd.Series(" ".join(meaningful_words))

    # Generate predictions
    pred = pipe_tvec_nb.predict(final_text)[0]

    if pred == 1:
        output = "Negative"
    else:
        output = "Postive"

    return output

samplestr = 'Ampanget ng quality'
result = predict(samplestr)
print(f'RAW_TEXT: {samplestr}')
print(F'SENTIMENT: {result}')

#tvec_nb_pred = pipe_tvec_nb.predict(X_val)

# Print accuracy scores
#train_score = (pipe_tvec_nb.score(X_train, y_train) * 100) 
#val_score = (pipe_tvec_nb.score(X_val, y_val) * 100)
#print('Training score:', "{:.2f}".format(train_score))
#print('Validation score:', "{:.2f}".format(val_score))
#print('')

# Print classification report and confusion matrix
#cmat(y_val, tvec_nb_pred, 'validation set')

# Read test set into a dataframe
#test = pd.read_csv("C:/Users/acer/Desktop/python_sht/Sentiment-Analysis-for-Shopee/data/clean_test.csv")
#test = pd.read_csv("C:/Users/acer/Desktop/python_sht/Sentiment-Analysis-for-Shopee/data/shopee_reviews.csv")
#print(test.head())
#X_test = test['content_stem']
#y_test = test['target']
#test_pred = pipe_tvec_nb.predict(X_test)

#test_score = (accuracy_score(y_test, test_pred) * 100)
#print('Evaluation metrics for test set')
#print('')
#print('Accuracy score: ', "{:.2f}".format(test_score))
#print('')