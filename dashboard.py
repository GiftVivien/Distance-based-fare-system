import pandas as pd
import streamlit as st
from keywords_detection import detect_keywords_in_comment, keyword_counts, keyword_counts_graph, plot_keyword_sentiment_distribution
from sentiment_distribution import sentiment_trend_yearly, plot_sentiment
from MLsentimentPrediction import train_sentiment_model, predict_sentiment
import joblib

# Streamlit app setup
st.title('Public Sentiment Analysis Dashboard')
st.write("This dashboard allows you to visualize and analyze public sentiment regarding the new transport fare model.")

# File uploader for user to upload their dataset
uploaded_file = st.file_uploader(r"C:\Users\USER\Downloads\Documents\Data Analytics Projects\Distance-based Fare\cleaned_file.csv", type=["csv"])
if uploaded_file:
    cleaned_file = pd.read_csv(uploaded_file)
    st.write("Data loaded successfully")
    
    # Show first few rows of the dataset
    st.dataframe(cleaned_file.head())

    # Detect keywords
    cleaned_file = detect_keywords_in_comment(cleaned_file, text_column='comment')

    # Save cleaned data with keywords
    cleaned_file.to_csv(r'C:\Users\USER\Downloads\Documents\Data Analytics Projects\Distance-based Fare\cleaned_Keywords_file', index=False)

    #sentiment distribution graph
    plot_sentiment(cleaned_file)

    # Keyword counts and plot
    counts = keyword_counts(cleaned_file, none_label="Uncategorized")
    keyword_distribution = keyword_counts_graph(counts)

    # Exploding keywords and plotting sentiment distribution
    df_exploded = cleaned_file.copy()
    df_exploded['keywords'] = df_exploded['keywords'].str.split(', ')
    df_exploded = df_exploded.explode('keywords')
    plot_keyword_sentiment_distribution(df_exploded)

   
    # Sentiment trend over the years
    sentiment_trend_yearly(cleaned_file)

    # Train sentiment prediction model if not already saved
    model_path = r'C:\Users\USER\Downloads\Documents\Data Analytics Projects\Distance-based Fare\sentiment_model.pkl'
    vectorizer_path = r'C:\Users\USER\Downloads\Documents\Data Analytics Projects\Distance-based Fare\tfidf_vectorizer.pkl'
    
    # Check if model already exists, if not, train it
    try:
        model = joblib.load(model_path)
        tfidf = joblib.load(vectorizer_path)
        st.write("Sentiment model loaded successfully.")
    except:
        st.write("Model not found. Training a new model...")
        train_sentiment_model(cleaned_file, model_path=model_path, vectorizer_path=vectorizer_path)

    # Sample texts for sentiment prediction
    sample_texts = [
        "This transport fare system is very fair and efficient.",
        "I hate the new pricing model. It's terrible.",
        "It's okay, not much different than before."
    ]

    # Perform sentiment prediction
    results = predict_sentiment(sample_texts, model_path=model_path, vectorizer_path=vectorizer_path)
    
    # Display prediction results
    st.write("Sentiment Prediction Results for Sample Texts:")
    st.write(results)

    # Save the prediction results to CSV
    prediction_results = pd.DataFrame({'Text': sample_texts, 'Sentiment': results})
    prediction_results.to_csv(r'C:\Users\USER\Downloads\Documents\Data Analytics Projects\Distance-based Fare\sentiment_predictions.csv', index=False)
    st.write("Sentiment predictions saved to CSV.")
