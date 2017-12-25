# Twitter_Word2Vec
Twitter-based traffic information system based on vector representations for words

In this project, I am using vector representations for words (i.e., word2vec models) to classify tweets into two groups: 1) Non-traffic-
related, 2) Traffic-related. 

Bag-of-words representation is a common method in literature for tweet modeling and retrieving traffic information, yet it suffers from 
the curse of dimensionality and sparsity. To address these issues, our specific objective is to propose a simple and robust framework on
the top of word embedding for distinguishing traffic-related tweets against non-traffic-related ones. In our proposed model, a tweet is 
classified as traffic-related if the semantic similarity between its words and a small set of traffic keywords exceeds a threshold value.
Semantic similarity between words is captured by means of word-embedding models, which is an unsupervised learning tool. The proposed 
model is as simple as having only one parameter, threshold, which does not require a large training set for computing its optimal value.
Although the model needs a few traffic keywords (i.e., around 10) as input, it achieves a high-quality performance with any traffic word
set regardless of the number, source, and frequency of selected keywords. 

In addition to addressing the shortcomings in traditional bag-of-words representation, our proposed model takes advantage of outstanding
merits including: 1) Model is primarily contingent on unsupervised learning tools (i.e., word2vec models). 2) Model has only one trainable
parameter (i.e., Threshold). 3) Model needs only a small set of traffic keywords (i.e., around 10), while it is insensitive to the number,
source, and frequency of traffic keywords. 3) Model does not require any assumption to be made on distribution of training data and other
models’ variables. 4) Model does not need a large training set for neither extracting traffic keywords nor training the Threshold
parameter. 5) Model is capable of detecting TR tweets that have no words in the selected traffic keywords. 6) Model takes all tweet’s 
words into account for the classification task. Furthermore, the reported indexes such as test accuracy, precision, recall, and F-score 
reveal the superiority of the proposed model in attaining high-prediction quality. As an application, the proposed framework can be used
by traffic management centers as a complementary source for real-time monitoring of traffic conditions. 

In this repository, the codes for collecting and labeling 51,100 tweets as well as training the classification are provided. 


