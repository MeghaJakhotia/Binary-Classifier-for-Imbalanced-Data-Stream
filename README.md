# Binary-Classifier-for-Imbalanced-Data-Stream
A binary classifier model to be run on two different datasets

## Motivation
Most of the Classifiers in supervised learning predict accurately for balanced data sets. But when dealt with imbalanced data set, it intends to be biased towards majority class thus ignoring the minority class. Learning from imbalanced data has conventionally been conducted on stationary data sets. Recently, there have been several methods proposed for mining imbalanced data streams. Training data is read in consecutive chunks, each of which is considered as a conventional imbalanced data set called data streams. We worked on the problem of classifying imbalanced data streams in binary classifiers. 

## Strategy
Dealing with imbalanced datasets entails strategies such as improving classification algorithms or balancing classes in the training data (data pre-processing) before providing the data as input to the machine learning algorithm.

Many practical classification tasks involve dealing with imbalanced class distributions, such as detection of fraudulent credit card transactions and diagnosis of rare diseases. Therefore, sampling methods can easily be performed to make data chunks class-balanced, such as over-sampling by generating synthetic minority class instances, oversampling by reusing historical training instances, and under-sampling by clustering the majority class.

The **_issues_** in imbalanced data stream contain the non availability of class labels, the emergence of new classes according to the input data and also the fading of existing classes. 

We found a method via which we will classify the data so that it belongs to the appropriate class by not being biased towards the majority class and not considering the minority class as noise.

## Technologies
Used the programming language R and different machine learning models such as Naive Bayes, Ozaboost, Clustering and Sampling techniques.

## Installation
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
1. Download the Github repo
2. Download R and different machine learning packages used in the project on your local machine. 
3. Run any of the code. The data is provided in the txt files in the same folder.

## Authors
- [Megha Jakhotia](https://github.com/MeghaJakhotia)
- [Nikita Luthra](https://github.com/nikitaluthra)
- [Yash Kapadia](https://github.com/yashkapadia)

```
Special Thanks to our Prof - Dr. Kiran Bhowmick at DJ Sanghvi College of Engineering, Mumbai, India.
```

