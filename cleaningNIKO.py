#jeg indlæser først modulerne jeg måske kommer til at bruge
import re
import pandas as pd 
import matplotlib.pyplot as plt

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


print(my_cleaning_df['remove_special_chars'].head(10))

