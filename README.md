[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

# messageSA
A side project for experimenting with some NLP, classification, sentiment analysis

# Data Sources
This project uses two primary data sources. The [Standord IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and the [Sent140 twitter dataset](http://help.sentiment140.com/for-students) with 1.6 Million labeled tweets.

# Dependencies
This project uses:
  - NLTK (tokenizing, stemming)
  - Spacy (tokenizing, stemming)
  - SciKit (classification, metrics)
  - Yellowbrick (viz)
  
  
  # Results
  Results are mixed. Classifier struggles to break 72% F1 on the twitter dataset, and 85% on the IMDB dataset. Need to experiment with data cleaning, and hyperparamter tuning. 
  
  ![image](https://github.com/devin040/messageSA/blob/master/results/LRmetrics.png)
  ![image](https://github.com/devin040/messageSA/blob/master/results/cm.png)
  
