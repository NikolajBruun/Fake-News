#jeg indlæser først modulerne jeg måske kommer til at bruge
import re
import pandas as pd 
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from cleantext import clean 
import matplotlib.pyplot as plt
from collections import Counter 

# Sørg for, at nødvendige data er downloadet
nltk.download('punkt')
nltk.download('stopwords')


df = pd.DataFrame(pd.read_csv("TRAIN.csv"))


def all_lower(x):
    x= x.lower()
    return x

def remove_special_chars(x):
    return re.sub(r"[^a-zA-Z0-9<>]", " ", x)

auto_cleaning_df = df[["content","type"]].copy()

def sub_dates(x):
    sub_dates = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', ' <DATE> ', x)
    sub_dates = re.sub(r'\b[a-z]+\s\d{1,2},\s\d{4}\b', '<DATE>', sub_dates)
    sub_dates = re.sub(r'\b\d{1,2}[,]?\s[a-zA-Z]+\s\d{4}\b', '<DATE>', sub_dates)
    return sub_dates

'''def remove_special_chars(x):
    return re.sub(r"[^a-zA-Z0-9<>]", " ", x)
'''
stop_words = set(stopwords.words('english'))


ps=PorterStemmer()

def autocleandf(df):
    df['sub_dates']=df['content']
    df['cleaned']=df['sub_dates'].apply(
        lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_numbers=True, no_punct=True, no_currency_symbols=True)
    )
    df['tokens']=df['cleaned'].apply(lambda x: word_tokenize(x))
    df['filtered_tokens']=df['tokens'].apply(lambda x: [
    word for word in x if word.lower() not in stop_words and not any(tword in word.lower() for tword in ["number", "date", "url", "email", "<", ">"])
    ])
    df['stemmed_text'] = df['filtered_tokens'].apply(lambda words: " ".join(ps.stem(word) for word in words))
autocleandf(auto_cleaning_df)




samlet_tekst = ' '.join([' '.join(words) for words in auto_cleaning_df['filtered_tokens']]) #gør til lang string
alle_ord = samlet_tekst.split() #gør til liste

#gør raw content fil til en string



ord_antal=Counter(alle_ord)     


hypiggeste_ord = ord_antal.most_common(50)              #De 50 mest hyppige ord i 'ord_antal', laves til en liste mad tupler, 
                                                        #bestående af ord og ordet's hyppighed



category_mapping = {
    "fake": "Fake News",
    "satire": "Fake Fake",
    "bias": "Fake News",
    "conspiracy": "Fake News",
    "junksci": "Fake News",
    "hate": "Not Fake",
    "clickbait": "Not Fake",
    "unreliable": "Fake News",
    "political": "Not Fake",
    "reliable": "Not Fake"  
}

# Apply the mapping
auto_cleaning_df['broad_category'] = auto_cleaning_df['type'].map(category_mapping)

print(auto_cleaning_df[['stemmed_text','broad_category']].head(10))

auto_cleaning_df.to_csv("INDSÆTHERDANNY.csv", index=False)