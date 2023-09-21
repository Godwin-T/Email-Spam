# Import Libraries
import re
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load Data
def load_data(path):

    data = pd.read_csv(path)
    data.columns = data.columns.str.lower()
    return data

# Merge data
def merge_data(ling_data, evron_data):

    evron_data['label'] = (evron_data['category'] == 'spam').astype('int')
    evron_data = evron_data[['message', 'label']]

    ling_data = ling_data[['message', 'label']]
    data = pd.concat([evron_data, ling_data])
    return data

# Filter data
def filter_data(data):

    data_ham = data[data['label'] == 0]
    data_spam = data[data['label'] == 1]
    quater = data_ham.shape[0]//4

    data_ham = data_ham.iloc[:quater]
    data = pd.concat([data_ham, data_spam])
    data = data.sample(frac= 1)
    data = data.reset_index(drop =True)
    return data

# Remove stopwords
def remove_stopwords_alphanum(messages):
    new_message = []

    for message in messages:
        # clean_message = []
        # for sent in message:

        #     words = re.sub(r'[^a-zA-Z0-9\s]', '', sent)
        #     words = word_tokenize(words)
            
        #     filtered_words = [word for word in words if word not in stopwords.words('english')]
        #     filtered_sent  = ' '.join(filtered_words)
        #     clean_message.append(filtered_sent)
        # new_message.append(clean_message)

        words = re.sub(r'[^a-zA-Z0-9\s]', '', message)
        words = word_tokenize(words)
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        filtered_sent  = ' '.join(filtered_words)
        new_message.append(filtered_sent)

    return new_message

# Stematize and Lematize message
def stem_lemmatize(messages):

    port = PorterStemmer()
    lem = WordNetLemmatizer()

    new_message = []

    # for message in messages:
    #     clean_message = []
    #     for sent in message:
    #         words = word_tokenize(sent)
    #         words = [lem.lemmatize(port.stem(word)) for word in words]
    #         sent = ' '.join(words)
    #         clean_message.append(sent)
    #     new_message.append(clean_message)

    for message in messages:
        words = word_tokenize(message)
        words = [lem.lemmatize(port.stem(word)) for word in words]
        sent = ' '.join(words)
        new_message.append(sent)
        
    return new_message     

# Split and lransform data
def split_vetorise(label, messages):

    cv = CountVectorizer(ngram_range=(1,3), max_features=30)
    xtrain, xtest, ytrain, ytest = train_test_split(messages,
                                                    label, test_size=0.3, random_state=0)      
    
    xtrain_vec = cv.fit_transform(xtrain)
    xtest_vec = cv.transform(xtest)
    # xtrain_vec = [cv.fit_transform(message).toarray() for message in xtrain]
    # xtest_vec = [cv.transform(message) for message in xtest]
    output = (xtrain_vec, xtest_vec, ytrain, ytest)
    return output

# Data preprocessing pipeline
def preprocess(data):

    data['message'] = data['message'].str.lower()
    labels = data['label'].tolist()
    messages = data['message'].tolist()
    # messages = [sent_tokenize(message) for message in messages]
    messages = remove_stopwords_alphanum(messages)
    messages = stem_lemmatize(messages)
    output = split_vetorise(labels, messages)
    return output

def train(xtrain, ytrain):
    nb = MultinomialNB()
    nb.fit(xtrain, ytrain)
    return(nb)

def evaluate(message, label, nb):
    
    pred = nb.predict(message)
    acc = accuracy_score(label, pred)
    return acc
def main(ling_path = './data/Ling-spam/messages.csv', 
         evron_path = './data/evron/mail_data.csv'):
    
    ling_data = load_data(ling_path)
    evron_data = load_data(evron_path)
    data = merge_data(ling_data, evron_data)
    data = filter_data(data)
    (xtrain, xtest, ytrain, ytest) = preprocess(data)
    model = train(xtrain, ytrain)
    model_train_acc = evaluate(xtrain, ytrain, model)
    model_test_acc = evaluate(xtest, ytest, model)
    print(f' Model accuracy on the train data: {model_train_acc}')
    print(f'Model accuracy on the test data: {model_test_acc}')

main()