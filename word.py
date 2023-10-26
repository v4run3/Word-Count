import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download the 'punkt' resource
nltk.download('punkt')

# Ask the user for input text
text = input("Enter the text for word frequency analysis: ")

# Tokenize the text
words = word_tokenize(text)

# Convert words to lowercase and remove punctuation
words = [word.lower() for word in words if word.isalnum()]

# Remove stopwords (common words like 'the', 'and', 'in' that don't provide meaningful insights)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Calculate word frequency
word_freq = Counter(filtered_words)

# Create a DataFrame for word frequency analysis
word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])

# Sort the DataFrame by frequency in descending order
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

# Display the top 10 most frequent words
print("Top 10 Most Frequent Words:")
print(word_freq_df.head(10))

# Plot word frequency distribution (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(word_freq_df['Word'][:10], word_freq_df['Frequency'][:10])
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
