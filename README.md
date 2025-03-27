# Fake-News
Vores eksamensprojekt... try not to fuck it up

hvis i vil change lidt i den her kan i skrive:
notepad README.md (windows)
nano README.md (mac)

Husk at commit tho

test




This project include multiple Jypiter Notebooks. There are different notebooks for splitting the dataset, cleaning & stemming, logistic regression and the Simple vector model. Below is the step by step guide showing the order of execution, inputs and outputs. 


Chronological order of how our files should be run

- Cleaning_File.ipynb
- CSV_split.ipynb
- LogisticReg.ipynb
- SVM.ipynb


Graphs.ipynb
LengthCorr.ipynb
liardataset.ipynb


### Cleaning_File.ipynb
This jupyter notebook file pipelines the cleaning process of all datasets. The cleaning include, removing all articles that aren't english, cleaning and stemming. 

Input:
  995,000_rows.csv
  liar
  bbc_
  


### CSV_split.ipynb

Purpose is to divide the large corpus into a training set(80%), validation set(10%) and a test set(10%). 
Before we split we also divide the articles into a broad category of "Fake News" and "Reliable" which we will use in the logistic regression and SVM. 

Input:
  large_corpus_cleaned.csv
  
  995,000_rows.csv

Output:
  train_set.csv
  val_set.csv
  test_set.csv
  
  
  










