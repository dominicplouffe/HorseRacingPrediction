# Predicting Horse Racing Results
A few years ago I decided to take on a personal project to predict horse racing results. When I started, I didn’t really know exactly what the end goal of the project was suppose to be.  I was undertaking a journey to help become a better programmer and maybe learn a bit about machine learning as well.

Over the years I’ve worked on several different machine learning models to predict the outcome of a horse race. Here I’ll be talking about one approach that I’ve taken.  Hope you enjoy what I did and learn a few things along the way.

Most of you probably know horse racing in the traditional thoroughbred racing.  That is, jockey sitting on the horse that’s running along the track. Popular races like the Kentucky Derby, Preakness Stakes, and Belmont Stakes for example. I’ve grown up and watch a different type of horse racing, Harness Racing. These are a different type of horse that pull a sulky (two wheel cart) where the driver sits.

In general both types of racing are very similar but there are a few key differences:
- In thoroughbred racing, the horses start from a standstill whereas in harness racing the horses start from a running start.
- North American harness are all 1 mile races whereas thoroughbred races have many different lengths. (Note, harness race tracks are not all the same size, but the race is always one mile)
- Harness Racing horses don’t need as much time off between races as thoroughbred horses.
- Harness racing have two different running style. You have pacers and trotters. If you are in a trotting race the driver always needs to keep his horse in a trot.  If the horse breaks strides the driver must slow down until the horse is in last place before continuing to race.

The model that I’ll be talking about uses a SVM regression algorithm. Regression algorithm are nice for horse racing predictions. Based on a set of features (which are listed below), you teach an algorithm what types of features and values go with a horse that finishes in first place, second place, third and so on. When it comes to predictions, the algorithm can then estimate approximately which position a horse will come in based on the same type of feature set.

To run this demo, you will need:

- Python 2.7
- Pip
- Virtualenv
- Sklearn
- Scipy
- Numpy

## The Goal
How many times can I predict the winner of a horse race
## The Data
I’ve been gathering harness racing entries and results from across North America over the past 5 years. The data has come from public sources on the internet.

For this test, we will be training our model on data from December 1st 2016 to February 28th 2017.  Harness racing runs throughout the year and the weather has a lot to do in the outcome of the race.  Rain, snow, wind, etc affect how the horse. I have found that training the model on a longer time frame yield worse results.

The validation data that we will use will be from March 1st to March 31st 2017.

The data can be downloaded from here if you want to check it out for yourself.
## Features
Our model will be trained on 20 different features that I came up with.  The file that we are using in this test can be downloaded from here:
