#jeg indlæser først modulerne jeg måske kommer til at bruge
import re
import pandas as pd 
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Sørg for, at nødvendige data er downloadet
nltk.download('punkt')
nltk.download('stopwords')


df = pd.DataFrame(pd.read_csv("news_sample.csv"))

my_cleaning_df = df[["content"]].copy()


def all_lower(x):
    x= x.lower()
    return x

def remove_whitespace(x):
    remove= re.sub(r'\s+',' ',x).strip()
    return remove
def sub_num(x):
    sub_num = re.sub(r'\s\d*\s', ' <NUM> ',x)
    return sub_num
def sub_dates(x):
    sub_dates = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', ' <DATE> ', x)
    sub_dates = re.sub(r'\b[a-z]+\s\d{1,2},\s\d{4}\b', '<DATE>', sub_dates)
    sub_dates = re.sub(r'\b\s\d{1,2}\s[a-z]+\s\d{4}\b', '<DATE>', sub_dates)
    return sub_dates
def sub_email(x):
    sub_email=re.sub(r'\b[A-Za-z0-9.-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9]*\b', ' <EMAIL> ',x)
    return sub_email
def sub_url(x):
    sub_url =re.sub (r'http.?://[^\s]+',' <URL> ', x )
    return sub_url
def remove_special_chars(x):
    return re.sub(r"[^a-zA-Z0-9<>]", " ", x)


def clean_df(df,content):
    df['all_lower'] = df[content].apply(all_lower)
    df['remove_whitespace']=df['all_lower'].apply(remove_whitespace)
    df['sub_num'] = df['remove_whitespace'].apply(sub_num)
    df['sub_dates'] = df['sub_num'].apply(sub_dates)
    df['sub_email'] = df['sub_dates'].apply(sub_email)
    df['sub_url'] = df['sub_email'].apply(sub_url)
    df['remove_special_chars'] = df['sub_url'].apply(remove_special_chars)

clean_df(my_cleaning_df,'content')


auto_cleaning_df = df[["content"]].copy()

def sub_dates(x):
    sub_dates = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', ' <DATE> ', x)
    sub_dates = re.sub(r'\b[a-z]+\s\d{1,2},\s\d{4}\b', '<DATE>', sub_dates)
    sub_dates = re.sub(r'\b\d{1,2}[,]?\s[a-zA-Z]+\s\d{4}\b', '<DATE>', sub_dates)
    return sub_dates

#auto_cleaning_df['sub_dates'] = auto_cleaning_df['content'].apply(sub_dates)
def remove_special_chars(x):
    return re.sub(r"[^a-zA-Z0-9<>]", " ", x)


#auto_cleaning_df['clean_with_specchars'] = auto_cleaning_df['sub_dates'].apply(
    #lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_numbers=True)
#auto_cleaning_df['cleaned'] = auto_cleaning_df['clean_with_specchars'].apply(remove_special_chars)
# Tokenisering
#auto_cleaning_df['tokens'] = auto_cleaning_df['cleaned'].apply(lambda x: word_tokenize(x))
# Stopwords fra NLTK
stop_words = set(stopwords.words('english'))

# Betingelse for at fjerne stopwords og de ord, vi ikke ønsker
#auto_cleaning_df['filtered_tokens'] = auto_cleaning_df['tokens'].apply(lambda x: [
  #  word for word in x if word.lower() not in stop_words and not any(tword in word.lower() for tword in ["number", "date", "url", "email", "<", ">"])
#])

def autocleandf(df):
    df['sub_dates']=df['content']
    df['clean_with_specchars']=df['sub_dates'].apply(
        lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_numbers=True)
    )
    df['cleaned']=df['clean_with_specchars'].apply(remove_special_chars)
    df['tokens']=df['cleaned'].apply(lambda x: word_tokenize(x))
    df['filtered_tokens']=df['tokens'].apply(lambda x: [
    word for word in x if word.lower() not in stop_words and not any(tword in word.lower() for tword in ["number", "date", "url", "email", "<", ">"])
    ])
autocleandf(auto_cleaning_df)
#print("jegprinterher", auto_cleaning_df['filtered_tokens'].head())


samlet_tekst = ' '.join([' '.join(words) for words in auto_cleaning_df['filtered_tokens']]) #gør til lang string
alle_ord = samlet_tekst.split() #gør til liste
alltext = '\n'.join(df["content"]) #gør raw content fil til en string
words=alltext.split() #dernæst til lsite
print("antal uncleaned words",len(set(words)))



import matplotlib.pyplot as plt
from collections import Counter 

ord_antal=Counter(alle_ord)     
print("Total wordcount after cleaning:",len(ord_antal))

print("Procentsvis ændring:", (len(ord_antal)/len(set(words)))*100 )

hypiggeste_ord = ord_antal.most_common(50)              #De 50 mest hyppige ord i 'ord_antal', laves til en liste mad tupler, 
                                                        #bestående af ord og ordet's hyppighed

ord = [ord for ord, tælling in hypiggeste_ord]          #Laver en liste med ordene fra 'hypiggeste_ord'. 
                                                        #Her itereres gennem hver tuple og udtrækkes ordet.
antal = [tælling for ord, tælling in hypiggeste_ord]    #Laver en liste med antal forekomster fra 'hypiggeste_ord'. 
                                                        #Her itereres gennem hver tuple og udtrækkes antal forekomster.


#NOTE TO SELF LAV EVT PLOT DEL OM 200% CHAT

#Plottet
plt.figure(figsize=(15, 6))                             #Figurens højde og bredde
plt.bar(ord, antal, color = "hotpink")                  #'ord' og 'antal' sættes på x- og y-aksen i plottet
plt.xticks(rotation=45, ha='right', fontsize=11)        #Teksten på x-aksen roteres og tekststørrelsen vælges
plt.show() 

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
print("=================================================================================")
print("=================================================================================")
#STEMMING!!!!

ps=PorterStemmer()

auto_cleaning_df['stemmed_text']=auto_cleaning_df['cleaned'].apply(lambda x: ps.stem(x))

print(auto_cleaning_df['stemmed_text'].head(50))
