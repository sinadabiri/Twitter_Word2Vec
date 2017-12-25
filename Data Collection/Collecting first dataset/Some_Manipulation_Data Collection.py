from pymongo import MongoClient
import tweepy
import os
import csv
from sklearn.feature_extraction.text import CountVectorizer
os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\Codes')

# Connecting to MongoDB, creating twitter database, and tweets collection
client = MongoClient()
db = client['twitter']  # or access by attribute style rather than dictionary style: db = client.twitter
Initial_Coll = db['Initial_Coll']
temporary = db['temporary']
Manual_Labeled_15000 = db['Manual_Labeled_15000']

# Authentication process to Twitter API
auth = tweepy.OAuthHandler('F9av1j73aaAKKmLwSOw0wUUD8', 'ulslZZoPRVT4atRWvKIX4VQmFnQv3fodBdWT5q6Q2KKp1h5a2Q')
auth.set_access_token('3405607174-us6ym0neIg8vyglbVzISa7weoe68qEuqpsBcupi',
                      'nDUawu1zuv3VzUt1FGis4zc1nTN1vsiFLiaeMGpPLgZMw')
api = tweepy.API(auth, compression=False, wait_on_rate_limit=False)

'''''
# Mix two labeled files: new and random. The mixed csv file is also subjected to the fiter file. So, it may not exactly
# be the same as combination of the other two files. 
with open('1-15000-Tweets-For-Labeling-new.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    Tweets = []
    for row in reader:
        Tweets.append(row)

with open('1-15000-Tweets-For-Labeling-random.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        Tweets.append(row)

with open('15000_Final_Mixed_Labeled.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in Tweets:
        writer.writerow(tweet)
'''

with open('15000_Final_Mixed_Labeled.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    Tweets = []
    for row in reader:
        Tweets.append(row)

vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()
count = 0
for i in range(len(Tweets)):
    if len(analyze(Tweets[i][2])) < 4:
        count += 1

with open('15000_Final_Mixed_Labeled.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in Tweets:
        if len(analyze(tweet[2])) >= 4 and tweet[0] != '-1':
            writer.writerow(tweet)
