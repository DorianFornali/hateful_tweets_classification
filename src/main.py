import re as regex
import sys
import joblib
import numpy as np
import pandas as pd
import html
import nltk
import time
import gensim.downloader as api

from imblearn.over_sampling import RandomOverSampler, SMOTE
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


def preprocess_tweet(text):
    # Convert to lowercase
    text = text.lower()
    # Convert HTML characters
    text = html.unescape(text)
    # Remove special characters, URLs, and user mentions
    text = regex.sub(r'&', 'and', text)
    text = regex.sub(r'http\S+|www\S+|@[^\s]+', '', text)
    text = regex.sub(r'[^a-zA-Z\s]', '', text)
    text = stemming(text)
    return text


def to_csv(_df, filename):
    path = "data/" + filename + ".csv"
    _df.to_csv(path, index=False)
    if VERBOSE:
        print("Successfully saved " + filename + " to " + path)


def stemming(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def embedTweet(df, embeddingModel):
    # Tokenize the tweets
    df['tweet'] = df['tweet'].apply(word_tokenize)

    # Function to get the embedding for a word, handling out-of-vocabulary words
    def get_word_embedding(word):
        try:
            return embeddingModel[word]
        except KeyError:
            # If the word is not in the vocabulary, return a zero vector
            return np.zeros(embeddingModel.vector_size)

    # Function to embed a tweet by taking the mean of word embeddings
    def embed_single_tweet(tweet):
        embeddings = [get_word_embedding(word) for word in tweet]
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            # If the tweet is empty, return a zero vector
            return np.zeros(embeddingModel.vector_size)

    # Embed each tweet in the DataFrame
    df['tweet'] = df['tweet'].apply(embed_single_tweet)




if __name__ == '__main__':
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    VERBOSE = False
    TRAINING = False
    EMBEDDING = False
    INTERACTIVE = False

    args = sys.argv[1:]
    if "-v" in args:
        VERBOSE = True
    if "-t" in args:
        TRAINING = True
    if "-e" in args:
        EMBEDDING = True
    if "-i" in args:
        INTERACTIVE = True

    if(len(args) == 0):
        # No args, printing out basic info
        print("Usage: python main.py [options]")
        print("Options:")
        print("-v: verbose")
        print("-t: training | Trains the models on the data prepared by train_test_splitting.py")
        print("-e: embedding | Embeds the tweets using word embeddings, NOT WORKING")
        print("-i: interactive | Allows the user to enter tweets and get predictions")
        print("Example: python main.py -v -i")
        exit(0)

    # The train test splitting script simply uses the train_test_split function from sklearn
    # and saves the training and testing data to two separate files to ease the process
    with open("src/train_test_splitting.py") as file:
        exec(file.read())

    """
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    """


    if VERBOSE:
        print("Loading data...")

    # We load the training and testing data
    df_train = pd.read_csv('data/train_resampled_file.csv')
    df_test = pd.read_csv('data/test_file.csv')

    if VERBOSE:
        print("Successfully loaded data")

    # We preprocess it

    if VERBOSE:
        print("Preprocessing data...")

    df_train['tweet'] = df_train['tweet'].apply(preprocess_tweet)
    df_test['tweet'] = df_test['tweet'].apply(preprocess_tweet)

    if VERBOSE:
        print("Successfully preprocessed data")

    if(EMBEDDING):
        # Here we will embed the tweets using word embeddings
        # We first load an already trained model from an api
        if(VERBOSE):
            print("Loading embedding model...")
        embeddingModel = api.load("glove-twitter-25")
        embeddingModel.save("models/embeddingModel")

        print("preembedding size:", df_train.shape)
        print(df_train.head(5))

        if VERBOSE:
            print("Embedding tweets...")
        # We will now use the model to embed the tweets
        embedTweet(df_train, embeddingModel)
        embedTweet(df_test, embeddingModel)
        to_csv(df_train, "train_embedded")
        if VERBOSE:
            print("Successfully embedded tweets")
        # We test on a single tweet


        print("postembedding size:", df_train.shape)
        print(df_train.head(5))


    # We already split the training data and testing beforehand in two searate files
    X_train = df_train.drop('class', axis=1)
    y_train = df_train['class']

    X_test = df_test.drop('class', axis=1)
    y_test = df_test['class']



    # We will try three different models

    # Logistic Regression model
    logistic_regression_params = {
        'tfidf__max_features': [5000, 10000, 15000],
        'model__C': [0.1, 1, 10],
        'model__max_iter': [500, 1000, 1500]
    }
    LR_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


    # Random Forest Classifier model
    random_forest_params = {
        'tfidf__max_features': [1000, 2000, 3000],
        'model__n_estimators': [10, 20, 30],
        'model__max_depth': [None, 7, 8]
    }
    RF_model = RandomForestClassifier(random_state=RANDOM_STATE)

    # MLP Classifier model
    mlp_params = {
        'tfidf__max_features': [1200, 2400, 5000],
        'model__hidden_layer_sizes': [(10,), (20,), (10, 10)],
        'model__max_iter': [50, 100, 150]
    }
    MLP_model = MLPClassifier(max_iter=1000, random_state=RANDOM_STATE, solver="adam")

    # We will use pipelines to combine the TF-IDF vectorizer and the model
    LR_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', LR_model)
    ])
    LR_grid = GridSearchCV(LR_pipeline, logistic_regression_params, cv=3, n_jobs=-1)

    RF_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', RF_model)
    ])
    RF_grid = GridSearchCV(RF_pipeline, random_forest_params, cv=3, n_jobs=-1)

    MLP_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', MLP_model)
    ])
    MLP_grid = GridSearchCV(MLP_pipeline, mlp_params, cv=3, n_jobs=-1)


    if TRAINING:

        # We will fit the models to the training data
        if VERBOSE:
            print("\nFitting models to training data...")
        currentTime = time.time()
        LR_grid.fit(X_train['tweet'], y_train)
        print("Time taken to fit LR model:", time.time() - currentTime)
        currentTime = time.time()
        RF_grid.fit(X_train['tweet'], y_train)
        print("Time taken to fit RF model:", time.time() - currentTime)
        currentTime = time.time()
        MLP_grid.fit(X_train['tweet'], y_train)
        print("Time taken to fit MLP model:", time.time() - currentTime)
        if VERBOSE:
            print("Successfully fit models to training data")

    if(TRAINING):
        # We will now save the models
        if VERBOSE:
            print("\nSaving models...")
        joblib.dump(LR_grid, 'models/LR_pipeline.joblib')
        joblib.dump(RF_grid, 'models/RF_pipeline.joblib')
        joblib.dump(MLP_grid, 'models/MLP_pipeline.joblib')
        if VERBOSE:
            print("Successfully saved models")

    else:
        # We will now load the models
        if VERBOSE:
            print("Loading models...")
        LR_grid = joblib.load('models/LR_pipeline.joblib')
        RF_grid = joblib.load('models/RF_pipeline.joblib')
        MLP_grid = joblib.load('models/MLP_pipeline.joblib')
        if VERBOSE:
            print("Successfully loaded models")

    # We will now make predictions on the test data
    if VERBOSE:
        print("Making predictions on test data...")

    LR_predictions = LR_grid.predict(X_test['tweet'])
    RF_predictions = RF_grid.predict(X_test['tweet'])
    MLP_predictions = MLP_grid.predict(X_test['tweet'])

    if VERBOSE:
        print("Successfully made predictions on test data")

    # We will now evaluate the models

    accuracyLR = accuracy_score(y_test, LR_predictions)
    accuracyRF = accuracy_score(y_test, RF_predictions)
    accuracyMLP = accuracy_score(y_test, MLP_predictions)

    print("\nLogistic Regression Accuracy:", accuracyLR)
    print("Random Forest Accuracy:", accuracyRF)
    print("MLP Accuracy:", accuracyMLP)

    LRClassificationReport = classification_report(y_test, LR_predictions)
    RFClassificationReport = classification_report(y_test, RF_predictions)
    MLPClassificationReport = classification_report(y_test, MLP_predictions)

    # We will now save the classification reports
    if VERBOSE:
        print("\nSaving classification reports...")
    with open("reports/LRClassificationReport.txt", "w") as file:
        file.write(LRClassificationReport)
    with open("reports/RFClassificationReport.txt", "w") as file:
        file.write(RFClassificationReport)
    with open("reports/MLPClassificationReport.txt", "w") as file:
        file.write(MLPClassificationReport)
    if VERBOSE:
        print("Successfully saved classification reports")

    # Also saving the best parameters for each model
    if VERBOSE:
        print("\nSaving best parameters...")
    with open("reports/LRBestParameters.txt", "w") as text_file:
        text_file.write(str(LR_grid.best_params_))
    with open("reports/RFBestParameters.txt", "w") as text_file:
        text_file.write(str(RF_grid.best_params_))
    with open("reports/MLPBestParameters.txt", "w") as text_file:
        text_file.write(str(MLP_grid.best_params_))
    if VERBOSE:
        print("Successfully saved best parameters")

    cm_LR = confusion_matrix(y_test, LR_predictions)
    cm_RF = confusion_matrix(y_test, RF_predictions)
    cm_MLP = confusion_matrix(y_test, MLP_predictions)

    # Print confusion matrices
    print("\nConfusion Matrix - Logistic Regression:")
    print(cm_LR)

    print("\nConfusion Matrix - Random Forest:")
    print(cm_RF)

    print("\nConfusion Matrix - MLP:")
    print(cm_MLP)

    # Save confusion matrices
    if VERBOSE:
        print("\nSaving confusion matrices...")
    pd.DataFrame(cm_LR).to_csv("reports/ConfusionMatrix_LR.csv", index=False)
    pd.DataFrame(cm_RF).to_csv("reports/ConfusionMatrix_RF.csv", index=False)
    pd.DataFrame(cm_MLP).to_csv("reports/ConfusionMatrix_MLP.csv", index=False)
    if VERBOSE:
        print("Successfully saved confusion matrices")

    # We will now predict the class of an input

    if INTERACTIVE:
        # Interactive means we give the user the possibility to enter multiples tweets and get predictions
        while True:
            text = input("Enter a tweet: ")
            text = preprocess_tweet(text)

            if EMBEDDING:
                # We will now embed the tweet
                text = word_tokenize(text)
                embedTweet(pd.DataFrame([text], columns=["tweet"]), embeddingModel)

            print("Preprocessed text:", text)
            prediction = LR_grid.predict([text])
            print("Predicted class with LR:", prediction)
            prediction = RF_grid.predict([text])
            print("Predicted class with RF:", prediction)
            prediction = MLP_grid.predict([text])
            print("Predicted class with MLP:", prediction)
            print("Ctrl+C to exit")