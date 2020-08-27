# Spam-Mails-Prediction
>The objective is to Predict the mail is spam or Ham using Naive Bayes Machine Learning Algorithm.  
>	In this we used both **Pipeline and Without Pipeline approach**. 

## Approach
 - First Spam mails file in CSV Format is Read.
 - New columns spam whose value is 1 for all spam mails and 0 for all ham mails is added in the dataset.
 - Then the dataset is splitted into two parts using the columns message and spam: 75% for Training and 25% for Testing.
 - Text data message is converted to word count vectors using CountVectorizer.
 - Mutinomial Naive Bayes model is made and then trained by fitting the train dataset.
 - Instead of doing CountVectorizer and model making We can also use **Pipeline to help automate machine learning workflows**.
 - If we use pipeline we just have to feed CountVectorizer and MultinomialNB then they operate by enabling a sequence of 
	   data to be transformed and correlated together in a model that can be tested and evaluated to achieve an outcome.

## Result
      In my case Score without using Pipeline is 98.49% and with Pipeline 98.38%.
