import pandas as pd
from sklearn.model_selection import train_test_split

# Læs CSV-filen ind
df = pd.read_csv('news_sample.csv')

# Først splittes dataene i træningssæt (80%) og resten (20%)
train, temp = train_test_split(df, test_size=0.2, random_state=42)

# Dernæst splittes den resterende del (temp) i valideringssæt (50% af 20% = 10%) og testset (50% af 20% = 10%)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Se på størrelserne af de tre sæt
print(f"Training data: {len(train)} rækker")
print(f"Validation: {len(val)} rækker")
print(f"Test: {len(test)} rækker")

# Gem de opdelte data i nye CSV-filer (valgfrit)
train.to_csv('train_set.csv', index=False)
val.to_csv('val_set.csv', index=False)
test.to_csv('test_set.csv', index=False)


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
