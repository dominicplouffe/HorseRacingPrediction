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

The data can be downloaded from **here** if you want to check it out for yourself.
## Features
Our model will be trained on 20 different features that I came up with.  The file that we are using in this test can be downloaded from **here**:

| # | Row Name | Description |
| --- | --- | --- |
| A | Row Key | Unique Identifier for the row.  Easier to insert into a database this way |
| B | Race Id | The ID of the race |
| C | Morning Line / Odds | The odds for the horse on that day |
| D | Not Used | Value should always be 0 |
| E | Core Features | A list of 20 features which will be used to train our model(See below) |
| F | Position | The position the horse finished in |

### Description of Core Features
All features will have a value of -1, 0 or 1

| # | Row Name | Description |
| --- | --- | --- |
| 1 | Post | If the horse is in the post position 1, 2, 3, 4, or 5 “1” else “0” |
| 2 | Speed | If horse was in the top 2 finish speeds, “1” else “0” |
| 3 | Horse Win % | If horse’s win % is over 50%, “1” else “0” |
| 4 | Horse WPS % | If horse’s WPS % is over 60%, “1” else “0” |
| 5 | Horse ROI | If horse’s lifetime ROI for a $2 bet is over $2, “1” else “0” |
| 6 | Driver Win % | If driver’s win % is over 50%, “1” else “0” |
| 7 | Driver WPS % | If driver’s WPS % is over 60%, “1” else “0” |
| 8 | Driver ROI | If driver’s lifetime ROI for a $2 bet is over $2, “1” else “0” |
| 9 | Trainer Win % | If trainer’s win % is over 50%, “1” else “0” |
| 10 | Trainer WPS % | If trainer’s WPS % is over 60%, “1” else “0” |
| 11 | Trainer ROI | If trainer’s lifetime ROI for a $2 bet is over $2, “1” else “0” |
| 12 | Minimum Races | If horse has races more than 5 races “1” else “0” |
| 13 | Previous Break | If horse has broken strides in the last 2 races, “0” else “1” |
| 14 | Days Since Last Race | If horse has raced over the last 21 days, “1” else “0” |
| 15 | Same Track | If horse is racing on the same track as the previous race, “1” else “0” |
| 16 | Same Driver | If horse’s driver is the same as the previous race, “1” else “0” |
| 17 | Last Race Result | If the horse finished in first in the previous race, “1” else “0” |
| 18 | Last Race WPS | If the horse finished in a WPS position in the last race, “1” else “0” |
| 19 | Last Three Race | If the horse finished in first in the last 3 races, “1” else “0” |
| 20 | Purse | If the purse is the same as the last race, “0” if it is lower “-1”, else “1” |

## The Code

### Setup the code
```> sudo pip pip install virtualenv
> virtualenv .venv
> source .venv/bin/activate
> sudo pip install numpy
> sudo pip install scipy
> sudo pip pip install sklearn
```
