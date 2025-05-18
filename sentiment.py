from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# function to apply VADER sentiment analysis
def apply_sentiment_analysis(df, text_column = 'sentiment'):
  #initiliaze VADER analyzer
  analyzer = SentimentIntensityAnalyzer()
  #Ensure column with no null values and create copy to avoid modifying DataFrame
  df = df[df[text_column].notnull()].copy()

  #compute compound score for each entry
  df['sentiment_score'] = df[text_column].apply(lambda x: analyzer.polarity_scores(x)['compound'])
  #assign sentiment labels to scores respecting the threshold 
  df['sentiment_label'] = df['sentiment_score'].apply(
    lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
  )

  #return DataFrame with new sentiment score and labels
  return df