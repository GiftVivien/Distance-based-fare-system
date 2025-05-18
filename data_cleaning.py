import pandas as pd

# define a function to load and clean a csv file
def load_and_clean(filepath):
  #read the file with encoding fallback
  try:
    df = pd.read_csv(filepath, encoding= 'ISO-8859-1') # to read the (non_UTF-8) special characters in the file

  except Exception as e:
    #print error message 
    print(f"Error reading file: {e}")
    return None

  #clean column names and convert to lowercase
  df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

  #convert date column if available
  if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date']) 

  #remove duplicates
  df.drop_duplicates(inplace = True)

  #drop rpws with missing columns
  text_columns = ['date', 'platform', 'sources/url', 'comment', 'sentiment']
  for col in text_columns:
    if col in df.columns:
      df = df[df[col].notnull()]
  
  #finally print a brief info to showcase what's in the file
  print("Successfully loading and cleaning.")
  print(df.info())

  #return cleaned format of data in DataFrame
  return df

