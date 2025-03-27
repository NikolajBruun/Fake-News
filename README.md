# Fake-News
Vores eksamensprojekt... try not to fuck it up

hvis i vil change lidt i den her kan i skrive:
notepad README.md (windows)
nano README.md (mac)

Husk at commit tho

test


## About the Project
The project consists of:
- A report of our findings
- Multiple Jyputer Notebooks.
  - Cleaning_File.ipynb
  - CSV_split.ipynb
  - Graphs.ipynb
  - LengthCorr.ipynb
  - Liardataset.ipynb
  - LogisticReg.ipynb
  - SVM.ipynb



Each Jyputer Notebook takes one or more of the datasets, and we will make sure to explain in depth, what the input and output of each file is. 


Below is a step by step guide for how we executed our code.


## Chronological order of how our files should be run

- Cleaning_File.ipynb
- CSV_split.ipynb
- LogisticReg.ipynb
- SVM.ipynb


### Notebooks for extra visualisation and analysis
Graphs.ipynb
LengthCorr.ipynb
liardataset.ipynb


## Cleaning_File.ipynb
This notebook runs the full cleaning pipeline. This includes removing all non-english articles, cleaning- and stemming the content column. 

This jupyter notebook the cleaning process of all datasets. The cleaning include, removing all articles that aren't english, cleaning and stemming. 

### Modules
- pandas
- re
- langdetech - detect
- nltk.tokenize - word_tokenize
- nltk.corpus - stopwords
- nltk.stem - PorterStemmer
- cleantext - clean 
- collections - Coutner
- pandarallel - pandarallel

### Input:
- 995,000_rows.csv
- bbc_scrape_uncleaned.csv
- liar_uncleaned.csv


### Output:
- large_corpus_cleaned.csv
- bbc_cleaned.csv
- liar_cleaned.csv
  


## CSV_split.ipynb
Purpose is to divide the large corpus into a training set(80%), validation set(10%) and a test set(10%). 
Before we split we also divide the articles into a broad category of "Fake News" and "Reliable" which we will use in the logistic regression and SVM. 

### Modules
- pandas 
- sklearn.model_selection - train_test_split

### Input:
- large_corpus_cleaned.csv
- bbc_cleaned.csv
- liar_cleaned.csv
  
  995,000_rows.csv

### Output:
  train_set.csv
  val_set.csv
  test_set.csv
  
  
  










