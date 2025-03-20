import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Indlæs data
df = (pd.read_csv("995,000_rows.csv"))
#995,000_rows

'''filtered_data = df.dropna(subset=["tags"])

# Split tags if they are stored as comma-separated strings
tag_list = []
for tags in filtered_data["tags"]:
    tag_list.extend(tags.split(','))  # Splitting multiple tags

# Count occurrences of each tag
tag_counts = Counter(tag_list)

# Get the top 10 most used tags
most_common_tags = tag_counts.most_common(25)

# Convert to separate lists for plotting
tags, counts = zip(*most_common_tags)

# Plot the data
plt.figure(figsize=(10, 6))
plt.barh(tags, counts, color='turquoise')
plt.xlabel("Frequency")
plt.ylabel("Tags")
plt.title("Top 25 Most Used Tags")
plt.gca().invert_yaxis()  # Invert to show the highest count on top
plt.show()'''


'''
# Fjern NaN-værdier i 'type'
unique_types = df['type'].dropna().unique()

# Beregn procentdelen for hver type
percentages = []
counts = []

for t in unique_types:
    total_count = len(df[df['type'] == t])  # Samlet antal artikler i denne kategori
    nan_count = len(df[(df['authors'].isna()) & (df['type'] == t)])  # Antal med NaN authors i denne kategori

    percent = (nan_count / total_count) * 100 if total_count > 0 else 0  # Undgå division med nul
    percentages.append(percent)
    counts.append(total_count)  # Gem det totale antal for visning

    print(f"{t}: {percent:.2f}% (Total: {total_count})")

# Plot bar-chart
plt.figure(figsize=(15, 6))
plt.bar(unique_types.astype(str), percentages, color ="turquoise")
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Article Types Based on 'authors' being 'NaN'")
plt.xlabel("Type")
plt.ylabel("Percent (%)")
plt.show()'''


df['content_length'] = df['content'].astype(str).apply(len)


# Loop gennem unikke 'type' værdier
for article_type in df['type'].unique():
    # Filtrer artikler med den specifikke type
    type_df = df[df['type'] == article_type]
    
    # Beregn gennemsnittet og medianen af 'content_length' for denne type
    average_length = type_df['content_length'].mean()
    median_length = type_df['content_length'].median()
    
    # Print resultaterne
    print(f"Type: {article_type}")
    print(f"Average content length: {average_length}")
    print(f"Median content length: {median_length}")
    print('-' * 40)  # Adskiller de forskellige typer for klarhed
    
total_average=df['content_length'].mean()
total_median=df['content_length'].median()
print(total_average)
print(total_median)
