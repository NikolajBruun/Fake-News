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


df = pd.DataFrame(pd.read_csv("news_sample.csv"))


my_cleaning_df = df[["content"]].copy()


def all_lower(x):
    x= x.lower()
    return x
def remove_special_chars(x):
    return re.sub(r"[^a-zA-Z0-9<>]", " ", x)

auto_cleaning_df = df[["content"]].copy()

def sub_dates(x):
    sub_dates = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', ' <DATE> ', x)
    sub_dates = re.sub(r'\b[a-z]+\s\d{1,2},\s\d{4}\b', '<DATE>', sub_dates)
    sub_dates = re.sub(r'\b\d{1,2}[,]?\s[a-zA-Z]+\s\d{4}\b', '<DATE>', sub_dates)
    return sub_dates

def remove_special_chars(x):
    return re.sub(r"[^a-zA-Z0-9<>]", " ", x)

stop_words = set(stopwords.words('english'))


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

samlet_tekst = ' '.join([' '.join(words) for words in auto_cleaning_df['filtered_tokens']]) #gør til lang string
alle_ord = samlet_tekst.split() #gør til liste
alltext = '\n'.join(df["content"]) #gør raw content fil til en string
words=alltext.split() #dernæst til lsite
print("antal uncleaned words",len(set(words)))

ord_antal=Counter(alle_ord)     
print("Total wordcount after cleaning:",len(ord_antal))

print("Procentsvis ændring:", (len(ord_antal)/len(set(words)))*100 )

hypiggeste_ord = ord_antal.most_common(50)              #De 50 mest hyppige ord i 'ord_antal', laves til en liste mad tupler, 
                                                        #bestående af ord og ordet's hyppighed

ord = [ord for ord, tælling in hypiggeste_ord]          #Laver en liste med ordene fra 'hypiggeste_ord'. 
                                                        #Her itereres gennem hver tuple og udtrækkes ordet.
antal = [tælling for ord, tælling in hypiggeste_ord]    #Laver en liste med antal forekomster fra 'hypiggeste_ord'. 
                                                        #Her itereres gennem hver tuple og udtrækkes antal forekomster.


#Nnote TO SELF LAV EVT PLOT DEL OM 200% CHAT

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

print(auto_cleaning_df['stemmed_text'].head(3))

category_mapping = {
    "fake": "Fake News",
    "satire": "Fake Fake",
    "bias": "Fake News",
    "conspiracy": "Fake News",
    "junksci": "Fake News",
    "hate": "Not Fake",
    "clickbait": "Not Fake",
    "unreliable": "Not News",
    "political": "Not Fake",
    "reliable": "Not Fake"  
}

# Apply the mapping
df['broad_category'] = df['type'].map(category_mapping)

# Check the result to make sure mapping is performed correctly
#print(df[['broad_category', 'type']].head(10))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
vectorizer = ord_antal.most_common(10000)

train = pd.read_csv("train_set.csv")
val = pd.read_csv('val_set.csv')
test = pd.read_csv('test_set.csv')

x_train=vectorizer.fit_transform(train['stemmed_text'])
x_val=vectorizer.fit_transform(val['stemmed_text'])
x_test=vectorizer.fit_transform(test['stemmed_text'])

y_train=train['broad_category']
y_val=val['broad_category']
y_test=test['broad_category']

logr=LogisticRegression(max_iter=1000) #spørg chat

logr.fit(x_train,y_train)




from sklearn.metrics import classification_report, f1_score
y_pred=logr.predict(x_train)