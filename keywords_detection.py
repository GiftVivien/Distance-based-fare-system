import re # built-in python function for pattern matching in text
import pandas as pd
import matplotlib.pyplot as plt
# Check if running inside Streamlit
try:
    import streamlit as st
    USING_STREAMLIT = True
except ImportError:
    USING_STREAMLIT = False
import seaborn as sns

#Dictionary with some keywords
keyword_dictionary = {
    "pricing": ["expensive", "cheap", "cost", "fare", "price", "charges", "refund balance solved"],
    "distance": ["distance", "long", "short", "journey", "trip"],
    "confusion": ["confusing", "don't understand", "unclear", "how", "why", "?", "misinformation"],
    "fairness": ["fair", "unfair", "equal", "justice", "biased"],
    "support": ["good", "better", "great", "like", "approve", "support", "best"],
    "complaint": ["hate", "bad","worst", "angry", "problem", "failure", "iliteracy", "issues", "malfunctions"],
    "utilization":["easy", "reliable"]

}

#function to extract keywords from texts
def extract_keywords(text, keyword_dictionary):
  found = set() #to store categories found in text
  if isinstance(text, str): #check if input is string
    text = text.lower() # convert to lowercase
    #loop over each category and keyword
    for category, words in keyword_dictionary.items():
      for word in words:
        if re.search(r'\b' + re.escape(word) + r'\b', text):
          found.add(category)

  #return categories else none if no matched
  return ', '.join(sorted(found)) if found else 'none'


#function to apply keyword extraction
def detect_keywords_in_comment(df, text_column, new_column="keywords", keyword_dict=keyword_dictionary):
  df = df.copy()
  df.columns = df.columns.str.strip().str.lower() #clean column names
  #Error if column not found
  if text_column.lower() not in df.columns:
    raise KeyError(f"'{text_column}' column not found in DataFrame")
  #store results in a new column
  df[new_column] = df[text_column.lower()].apply(lambda x: extract_keywords(x, keyword_dict))
  return df


#function to count keywords
def keyword_counts(df, none_label = "No keywords"):
  df = df.copy()
  df['keywords'] = df['keywords'].replace('none', none_label) #replace none with none_label
  counts = df['keywords'].value_counts()
  return counts  

#function to plot keyword frequency
def keyword_counts_graph(counts):
  plt.figure(figsize=(10, 5))
  sns.barplot(x = counts.index, y=counts.values, hue=counts.index,  palette='viridis', legend= False)
  plt.title('Keyword Catgories Frequency')
  plt.xlabel('Keyword Categories')
  plt.ylabel("Frequency")
  if USING_STREAMLIT:
    st.pyplot(plt)
  else:
    plt.show()

  

#function to plot sentiment distribution across keyword categories
def plot_keyword_sentiment_distribution(df):
    
    if 'sentiment' in df.columns:
      df['sentiment_label'] = df['sentiment'].str.lower()

    #define order and color
    sentiment_order = ['positive', 'neutral', 'negative']
    color_palette = {'positive': 'green', 'neutral': 'gold', 'negative': 'red'}

    # Duplicate keywords into separate rows if comma-separated
    df = df.copy()
    df['keywords'] = df['keywords'].str.split(', ')
    df = df.explode('keywords')

    # Clean keyword label if needed
    df['keywords'] = df['keywords'].str.strip().replace('', 'Uncategorized')

    # Convert sentiment column to categorical for consistent order
    df['sentiment_label'] = pd.Categorical(df['sentiment_label'].str.lower(), categories=sentiment_order, ordered=True)

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='keywords', hue='sentiment_label', palette= color_palette, hue_order=sentiment_order)
    plt.title("Sentiment Distribution per Keyword Category")
    plt.xlabel("Keyword Category")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.tight_layout()
    if USING_STREAMLIT:
      st.pyplot(plt)
    else:
      plt.show()


