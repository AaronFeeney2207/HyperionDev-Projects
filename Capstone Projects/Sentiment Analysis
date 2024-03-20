# 1. Load the en_core_web_sm model from spacy to use NLP
# 2. Preprocess the data - Remove stopwords and clean text for analysis.
# 2.1. Single out the review.text column - = df['review.text']
# 2.2. Use the dropna() function to extract missing values that are in this column - = df.dropna(subset=[review.text'])
# 3. Create a function that takes the product review in as an input and predicts its sentiment 
# 4. Test the sentiment analysis function on a few sample product reviews to verify its accuracy in predicting sentiment.

# 5. Write a brief summary report in a PDF file - sentiment_analysis_report.pdf.
# 5.1. Must Include:
# A description of the dataset used
# Details of the proprocessing setps 
# Evaluation of results 
# Insights into the models strengths and limitations 

# Tips:
# To help remove stopwords, utilise .is_stop in spacy - helps idenitfy stopwords in a sentence/string. 
# Make use of .lower(), .strip() and str() to perform basic text cleansing.
# Use the .sentiment attribute on spacy to analyse the review and determine whether it expresses a positive, negative or neutral sentiment. 
# To use the .polarity attribute install TextBlob Library - install spacytextblob
# TextBlob needs additional data before getting started, download the data with: python -m textblob.download_corpora
# Using the polairty attribute:
# polarity = doc._.blob.polarity
# Using sentiment attribute 
# sentiment = doc._.blob.sentiment
# .polarity attribute allows you to measure the strength of a sentiment ina product review.
# Score of 1 = very positive sentiment. Score fo -1 = a very negative sentiment. Score fo 0 = Neutral sentiment
# Can use .similairty to compare the similairty of two product reviews.
# Choose two product reviews from 'review.text' column and compare their similairty.

import pandas as pd
import numpy as np
import spacy 
from textblob import TextBlob


# Preprocessing data function
nlp = spacy.load('en_core_web_md')

def preprocess_data(data):

    # Processing data 
    doc = nlp(data.lower().strip())  
    # Removal of stopwords
    processed = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    return ' '.join(processed)

# Polarity value function 
def polarity_value(data):
    # Retrieve sentiment with TextBlob
    blob = TextBlob(data)
    polarity_score = blob.sentiment.polarity
    
    return polarity_score

# Sentiment Function 
def sentiment_value(df,review_data):
    polarity = 0
    chosen_review = df[review_data]
    polarity_score = polarity_value(chosen_review)

    if polarity_score > 0:
            polarity = 'positive'

    elif polarity_score < 0:
            polarity = 'negative'

    else:
            polarity = 'neutral'

    return print(f'The sentiment of your random review choice is: {polarity}')
        

# Putting dataset into a dataframe 
df = pd.read_csv('amazon_product_reviews.csv')

# Indexing our dataframe to single out our 'reviews.text' column to preprocess
df_reviews = df["reviews.text"]

# Sanity Check
print(df_reviews)

df_miss_val = df_reviews.isnull().sum()
print(df_miss_val)  # No missing values in this column so no need for dropna() fucntion to remove NaN values.


# Running data through our processing function with apply() method which applys the function to all data in each row in our 'reviews.text' column
df_cleaned_reviews = df_reviews.apply(preprocess_data)


# Processed data Sanity Check
print(df_cleaned_reviews.head())

# While loop to engage the user in helping us identify a random review to test our sentiment and polairty functions on single random reviews. 
while True:
    review_num = int(input('To obtain a random review from the dataset please pick a number between 0 and 4999: '))
    if review_num >= 0 and review_num <= 4999 :
        sentiment_value(df_cleaned_reviews, review_num)
        break
    else:
        print('Invalid input. Try again. ')


# Getting sentiment of all reviews with polairty function and putting all the data into a list.
sentiments = []

for data in df_cleaned_reviews:
    polarity_score = polarity_value(data)
    
    if polarity_score > 0:
        sentiment = 'positive'
    elif polarity_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    sentiments.append(sentiment)

print(sentiments)

# Percentage stats for the sentiments of all the reviews5001
positive_count = sentiments.count('positive')
negative_count = sentiments.count('negative')
neutral_count = sentiments.count('neutral')

total = len(sentiments)

positive_perc = (positive_count / total) * 100
negative_perc = (negative_count / total) * 100
neutral_perc = (neutral_count / total) * 100

print(f"Positive percentage: {positive_perc:.2f}%")
print(f"Negative percentage: {negative_perc:.2f}%")
print(f"Neutral percentage: {neutral_perc:.2f}%")

# Discovering the similarity of multiple reviews
# Going to compare review 98 with review 4, review 753 and review 165

review_to_compare = df_reviews[98]

# Cleaning of review to compare 
review_to_compare = nlp(review_to_compare)

review_list = []

# Using forloop to append my chosen reviews into one list together.
for reviews in df_reviews:
    if reviews == df_reviews[4]:
        review_list.append(reviews)
    elif reviews == df_reviews[753]:
        review_list.append(reviews)
    elif reviews == df_reviews[165]:
        review_list.append(reviews)
    else:
        continue

# Sanity check
print(review_list)

# For loop to determine similairty 
for reviews in review_list:
    similarity_score = nlp(reviews).similarity(review_to_compare)
    print(f"{reviews} - {similarity_score}")

# From this we can see that the 4th review is the most similar to mymodel review which is the 98th review as it has the score closest to 1 making it the 
# most similar. 








