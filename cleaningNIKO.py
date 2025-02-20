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

#print(df.head(2))

#jeg vælger kun at cleane 'content' kolonnen 

my_cleaning_df = df[["content"]].copy()

#print(my_cleaning_df.head(10))

def all_lower(x):
    if isinstance(x,str):
        x= x.lower()
    return x

my_cleaning_df['all_lower'] = my_cleaning_df['content'].apply(all_lower)

def remove_whitespace(x):
    remove= re.sub(r'\s+',' ',x).strip()
    return remove

my_cleaning_df['remove_whitespace'] = my_cleaning_df['all_lower'].apply(remove_whitespace)

def sub_num(x):
    sub_num = re.sub(r'\s\d*\s', ' <NUM> ',x)
    return sub_num

my_cleaning_df['sub_num'] = my_cleaning_df['remove_whitespace'].apply(sub_num)

def sub_dates(x):
    sub_dates = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', ' <DATE> ', x)
    sub_dates = re.sub(r'\b[a-z]+\s\d{1,2},\s\d{4}\b', '<DATE>', sub_dates)
    sub_dates = re.sub(r'\b\s\d{1,2}\s[a-z]+\s\d{4}\b', '<DATE>', sub_dates)
    return sub_dates
#sub dates needs to be at format DD-MM-YYYY, DD/MM/YYYY or YYYY-MM-DD (as i think this is the most common US-form)
my_cleaning_df['sub_dates'] = my_cleaning_df['sub_num'].apply(sub_dates)

def sub_email(x):
    sub_email=re.sub(r'\b[A-Za-z0-9.-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9]*\b', ' <EMAIL> ',x)
    return sub_email

my_cleaning_df['sub_email'] = my_cleaning_df['sub_dates'].apply(sub_email)

def sub_url(x):
    sub_url =re.sub (r'http.?://[^\s]+',' <URL> ', x )
    
    return sub_url

my_cleaning_df['sub_url'] = my_cleaning_df['sub_email'].apply(sub_email)


def remove_special_chars(x):
    return re.sub(r"[^a-zA-Z0-9<>]", " ", x)

my_cleaning_df['remove_special_chars'] = my_cleaning_df['sub_url'].apply(remove_special_chars)


#print(my_cleaning_df['remove_special_chars'].head(10))

#ovenfor har vi lidt hygge
#nu bruger jeg clean-text modulet til at cleane med... stoler vi nok lidt mere på tbh

from cleantext import clean 

auto_cleaning_df = df[["content"]].copy()



def sub_dates(x):
    sub_dates = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', ' <DATE> ', x)
    sub_dates = re.sub(r'\b[a-z]+\s\d{1,2},\s\d{4}\b', '<DATE>', sub_dates)
    sub_dates = re.sub(r'\b\d{1,2}[,]?\s[a-zA-Z]+\s\d{4}\b', '<DATE>', sub_dates)
    return sub_dates

auto_cleaning_df['sub_dates'] = auto_cleaning_df['content'].apply(sub_dates)
def remove_special_chars(x):
    return re.sub(r"[^a-zA-Z0-9<>]", " ", x)




#auto_cleaning_df['cleaned'] = clean(auto_cleaning_df['content'], lower=True, no_urls=True, no_emails=True, no_numbers=True)
auto_cleaning_df['clean_with_specchars'] = auto_cleaning_df['sub_dates'].apply(
    lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_numbers=True)
)


auto_cleaning_df['cleaned'] = auto_cleaning_df['clean_with_specchars'].apply(remove_special_chars)

#print(auto_cleaning_df['cleaned'].head(99))

#nice nu virker det

# Tokenisering
auto_cleaning_df['tokens'] = auto_cleaning_df['cleaned'].apply(lambda x: word_tokenize(x))

# Stopwords fra NLTK
stop_words = set(stopwords.words('english'))

# Betingelse for at fjerne stopwords og de ord, vi ikke ønsker
auto_cleaning_df['filtered_tokens'] = auto_cleaning_df['tokens'].apply(lambda x: [
    word for word in x if word.lower() not in stop_words and not any(tword in word.lower() for tword in ["number", "date", "url", "email", "<", ">"])
])

print(auto_cleaning_df['filtered_tokens'].head())

samlet_tekst = ' '.join([' '.join(words) for words in auto_cleaning_df['filtered_tokens']])

alle_ord = samlet_tekst.split()

alltext = '\n'.join(df["content"])
words=alltext.split()
print("antal uncleaned words",len(set(words)))

#print(samlet_tekst)
#print(alle_ord)


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

#Plottet
plt.figure(figsize=(15, 6))                             #Figurens højde og bredde
plt.bar(ord, antal, color = "hotpink")                  #'ord' og 'antal' sættes på x- og y-aksen i plottet
plt.xticks(rotation=45, ha='right', fontsize=11)        #Teksten på x-aksen roteres og tekststørrelsen vælges
#plt.show() 

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
print("=================================================================================")
#STEMMING!!!!
