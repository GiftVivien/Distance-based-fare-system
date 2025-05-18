from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_sentiment_model(dataframe, text_column='comment', label_column='sentiment',
                          model_path='sentiment_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    # Prepare features and labels
    X = dataframe[text_column]
    y = dataframe[label_column].str.lower()
    
    # Vectorize text data
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
    X_vec = tfidf.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


def predict_sentiment(texts, model_path='sentiment_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):

    import joblib
    import numpy as np

    # Load model and vectorizer
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)

    # Ensure input is a list
    if isinstance(texts, str):
        texts = [texts]

    # Transform text and predict
    X_vec = tfidf.transform(texts)
    predictions = model.predict(X_vec)

    return predictions.tolist()
