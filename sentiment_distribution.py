import seaborn as sns
import matplotlib.pyplot as plt
# Check if running inside Streamlit
try:
    import streamlit as st
    USING_STREAMLIT = True
except ImportError:
    USING_STREAMLIT = False
import pandas as pd

#define a function to plot sentiment distribution
def plot_sentiment(df):

  # custom colors for each sentiment
  custom_palette = {
    'positive': 'green',
    'neutral': 'gold',  
    'negative': 'red'
  }

  #define the order of sentiment
  order_xlabels =['positive', 'neutral', 'negative']
  plt.figure(figsize=(8, 5)) #set figure sizes

  #create count plot for sentiment
  df['sentiment'] = df['sentiment'].str.strip().str.lower()
  sns.countplot(x='sentiment', hue= 'sentiment', data= df, palette=custom_palette,order=order_xlabels, legend=False)
  plt.title("Public Sentiment Distribution")
  plt.xlabel("Sentiment")
  plt.ylabel("Number of Responses")

  if USING_STREAMLIT:
    st.pyplot(plt)
  else:
    plt.show()

#define function to plot sentiment trend over the years
def sentiment_trend_yearly(df):
  #convert date column to datetime
  df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
  #extract year from date column
  df = df.dropna(subset =['date']) # drop rows where conversion failed
  df['year'] = df['date'].dt.year
  #normalize to lowercase
  df['sentiment'] = df['sentiment'].str.lower()
  #group data by year and sentiment and count 
  sentiment_trend = df.groupby(['year', 'sentiment']).size().reset_index(name = 'count')

  # custom colors for sentiment types
  colors_map = {'positive': 'green', 'neutral': 'gold', 'negative': 'red'}

  #set figure size
  plt.figure(figsize=(10, 6))
  sns.barplot(data=sentiment_trend, x='year', y='count', hue='sentiment', palette=colors_map)

  plt.title('Sentiment Trend Over the Years')
  plt.xlabel('Year')
  plt.ylabel('Comments')
  plt.legend(title='Sentiment')
  plt.tight_layout()
  if USING_STREAMLIT:
    st.pyplot(plt)
  else:
    plt.show()






