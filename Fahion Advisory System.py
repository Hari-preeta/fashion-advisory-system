
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
import string

dataset = pd.read_csv("D:\Documents\Myntra Kurtis.csv")

#print(data.head())
#print(data.isnull().sum())

dataset = dataset.drop("Image",axis=1)
dataset = dataset.dropna()
#print(data.shape)
text = " ".join(i for i in dataset["Brand Name"])
sw= set(STOPWORDS)
wordcloud = WordCloud(stopwords=sw, 
                      background_color="violet").generate(text)
#print(wordcloud)

plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
highest_rated = dataset.sort_values(by=["Product Ratings"], 
                                 ascending=False)
highest_rated = highest_rated.head(5)
#print(highest_rated[['Product Info', "Product Ratings", "Brand Name"]])
mean = dataset['Product Ratings'].mean()
m = dataset['Number of ratings'].quantile(0.9)
n = dataset['Number of ratings']
a = dataset['Product Ratings']
dataset["Score"]  = (n/(n+m) * a) + (m/(m+n) * mean)

recommend = dataset.sort_values('Score', ascending=False)

#print(data.head(5))
x=int(input("order of prediction:"))
if (x==1):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].head(1))
    print(recommend['Product URL'].head(1))
elif (x==2):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[2])
    print(recommend['Product URL'].iloc[2])
elif (x==3):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[3])
    print(recommend['Product URL'].iloc[3])
elif (x==4):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[4])
    print(recommend['Product URL'].iloc[4])
elif (x==5):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[5])
    print(recommend['Product URL'].iloc[5])
elif (x==6):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[6])
    print(recommend['Product URL'].iloc[6])
elif (x==7):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[7])
    print(recommend['Product URL'].iloc[7])
elif (x==8):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[8])
    print(recommend['Product URL'].iloc[8])
elif (x==9):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[9])
    print(recommend['Product URL'].iloc[9])
elif(x==10):
    print(recommend[['Brand Name', 'Product Info',
                       'Product Ratings', 'Score', 
                       'Selling Price', 'Discount']].iloc[10])
    print(recommend['Product URL'].iloc[10])
