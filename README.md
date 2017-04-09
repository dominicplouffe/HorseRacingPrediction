# Predicting Horse Racing Results
A few years ago I decided to take on a personal project to predict horse racing results. When I started, I didn’t really know exactly what the end goal of the project was suppose to be.  My primary objective was to have a journey to help become a better programmer and maybe learn a bit about machine learning as well.  The result was [DeadHeat.ca](http://www.deadheat.ca).

Over the years I’ve worked on several different machine learning models to predict the outcome of a horse race. I've tried a variety of different flavours of classifiers, clustering engine and regression algorithms. Here I’ll be talking about one approach that I’ve taken.  Hope you enjoy what I did and learn a few things along the way.

---

Most of you probably know horse racing in the traditional [thoroughbred racing](https://en.wikipedia.org/wiki/Thoroughbred_horse_racing).  That is, jockey sitting on the horse that’s running along the track. Popular races like the Kentucky Derby, Preakness Stakes, and Belmont Stakes are thoroughbred races. I’ve grown up and watch a different type of horse racing, [harness racing](https://en.wikipedia.org/wiki/Harness_racing). These are a different type of horse that pull a sulky (two wheel cart) where the driver sits.

In general both types of racing are very similar but there are a few key differences:
- In thoroughbred racing, the horses start from a standstill whereas in harness racing the horses start from a running start.
- North American harness races all have the same distance, 1 mile, whereas thoroughbred races have many different lengths. (Note, harness race tracks are not all the same size, but the race is always 1 mile)
- Harness Racing horses don’t need as much time off between races as thoroughbred horses.
- Harness racing have two different running style. You have pacers and trotters. If you are in a trotting race the driver always needs to keep his horse in a trot.  If the horse breaks strides the driver must slow down until the horse is in last place before continuing to race.

---
## The algorithm

The model that we'll be creating will be using is a [Support Vector Maching regression](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) algorithm to train and predict results. Regression algorithm are nice for horse racing predictions. I used historical race data to create a set of features (which are listed below).  Features are a list of attributes (like which post the horse starts, the winning percentage of the horse, how good the driver is, etc..) that define the characteristics of a horse for a particular race. Using these features, you teach the algorithm the types of attributes a winning horse needs to have. 

When it comes to predictions, the algorithm can then estimate the position a horse will come in based on the same type of feature set.

## The Goal
The goal the algorithm will try to answer is: How many times can I predict the winner of a horse race?  A secondary goal is: How much money will I win or lose if I were to wager using the predictions made by the algorithm?

## The Data
I’ve been gathering harness racing entries and results from across North America over the past 5 years. The data has come from public sources on the internet.

For this test, we will be training our model on data from December 1st 2016 to February 28th 2017.  Harness racing runs throughout the year and the weather affects the outcome of the race.  Rain, snow, wind and temperature all affect how the race will go. I have found that training the model on a shorter time frame will yield to better results. Training with data close to the date you want to predict will, in general, have similar weather patterns.

The validation data that we will use to test the algorithm will be from March 1st to March 31st 2017.

The data can be downloaded from [here](https://www.dropbox.com/s/jxa01gtf20w71mh/raw_data.xlsx?dl=0) if you want to check it out for yourself.

## Features
Our model will be trained on 20 different features that I came up with.  Both the training and validation sets can be found in the github repo.

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

To run this demo, you will need:

- Python 2.7
- Pip
- Virtualenv
- Sklearn
- Scipy
- Numpy

### Setup the code
```$ git clone git@github.com:dominicplouffe/HorseRacingPrediction.git
$ sudo pip pip install virtualenv
$ virtualenv .venv
$ source .venv/bin/activate
$ sudo pip install numpy
$ sudo pip install scipy
$ sudo pip install sklearn
```
### Training the model
To train the model, we load training data, setup the training array (X) and target results (y). Each row is a result of a horse in a distinct race. The training array are the features described above and the target results is the finish position of the horse in that race.

Once we have all the rows formatted in a list we instantiate the regression algorithm and fit the model. Last, we save the model to a file so we can use it in a different class.

```python
def _get_data(self, filename):

    training_data = csv.reader(open('data/%s' % filename, 'rb'))

    logging.info('Training Finish Position')

    y = []  # Target to train on
    X = []  # Features


    for i, row in enumerate(training_data):
        # Skip the first row since it's the headers
        if i == 0:
            continue

        # Get the target
        y.append(float(row[-1]))

        # Get the features
        data = np.array(
            [float(_ if len(str(_)) > 0 else 0) for _ in row[5:-1]]
        )
        X.append(data.reshape(1, -1))

    return X, y

def train(self):

    clf = SVR(C=1.0, epsilon=0.1, cache_size=1000)
    X, y, = self._get_data('training_data.csv')

    # Fit the model
    clf.fit(X, y)

    # Pickle the model so we can save and reuse it
    s = pickle.dumps(clf)

    # Save the model to a file
    f = open('finish_pos.model', 'wb')
    f.write(s)
```

### Predictions and Validation
Once the model has been trained, we are now ready to validate how well it is working. 

To validate the results we iterate through the validation dataset, group every race together and using the same type of feature set as above, we predict the approximate position the horse will finish in.  To select the horse I think will win, we sort the predictions and pick the lowest value.

For example, a prediction value can be something like 1.45. This means that the model has fitted the features inside the first and second place markers. Essentially, the characteristics of the horse on this race is similar to a typical horse that finishes in first or second place.

```python
def predict(self):
    f = open('finish_pos.model', 'rb')
    clf = pickle.loads(f.read())
    f.close()

    validation_data = csv.reader(
        open('data/validation.csv', 'rb')
    )

    races = {}
    for i, row in enumerate(validation_data):
        if i == 0:
            continue

        race_id = row[1]
        finish_pos = float(row[-1])

        if race_id not in races:
            races[race_id] = []

        if finish_pos < 1:
            continue

        data = np.array([
            float(_ if len(str(_)) > 0 else 0)·
            for _ in row[5:-1]
        ])
        data = data.reshape(1, -1)
        races[race_id].append(
            {
                'data': data,
                'precition': None,
                'finish_pos': finish_pos·
            }
        )

    num_races = 0
    num_correct_pred_win = 0
    num_correct_pred_wps = 0
    for race_id, horses in races.iteritems():
        for horse in horses:
            horse['prediction'] = clf.predict(
                horse['data']
            )

        horses.sort(key=lambda x: x['prediction'])

        num_races += 1
        if horses[0]['finish_pos'] == 1:
            num_correct_pred_win += 1

        if horses[0]['finish_pos'] in [1, 2, 3]:
            num_correct_pred_wps += 1

    print('Number of races predicted => %s' % num_races)
    print('Number of correct win predictions = %s' % num_correct_pred_win)
    print('Number of correct WPS predictions = %s' % num_correct_pred_wps)
```
### Results
To validate the results I first ran a baseline for comparisson purposes. The baseline is the morning line (odds given by the race track for a horse before any wagers have been performed). The baseline is simple but will allow us to compare a non-algorithmic approach to my algorithm approach. The horse that was the favorite based on the morning line is assumed to have the better chance of winning. For example, a horse a 2 to 1 will have a better chance of winning compared to a horse with odds of 3 to 1. I picked the horse at 2 to 1 to win.

The validation file has a total amount of 2,896 races.  The favorite won 741 times (26%) and came in 1st, 2nd or 3rd 1666 (58%) of the time.

Now for our model. The regression algorithm fits the giving features to the curve that has been trained. The values to the prediction look something like 1.56, 3.90, etc.. Meaning that the features fit between a 1 and 2 or 3 and 4 respectively.  We loop through all the horses in a race, predict the outcome and sort on the prediction (lowest value is assumed to be winning).

The results are as follows:

Horses with the lowest prediction won 812 times (28%) and came in 1st, 2nd or 3rd 1820 times (63%).

<p align="center">
<img src="http://dplouffe.ca/static/img/baseline.png" border="0" />
</p>

The machine learning approach works slightly better.

### More Strategies
To make money wagering on horse racing you should not bet on all races. The approach a professional gambler takes is to analyze each race to try to find an advantage. A horse that is thought by the public to be a poor performer but that you see something positive is a horse you want to bet on. When I analyze race programs I only bet on races which I believe I can win, otherwise I move on to the next race. The challenge is to do this for every race each day is *impossible*! This is where algorithms come in really handy since they can analyze all races in a matter of seconds.

The next *attempts* uses a simplistic but affectinve approach to find an edge. That is, to try and find races which has a horse that's better than the rest.

This attempt simulates a bet when a horse with the lowest prediction has a whole position lower than the second horse. For example, if the lowest horse prediction is 1.20 and the second lowest is 1.90, I did not simulate a bet on the race. On the other hand, if a lowest horse prediction is 1.20 and the second lowest is 2.40, I simulated the bet.

The results are as follows:

<p align="center">
<img src="http://dplouffe.ca/static/img/offset_one.png" border="0" />
</p>

664 races met the above criteria. The horses with the lowest prediction won 251 times (38%) and came in 1st, 2nd or 3rd 477 times (72%). You can start to see that betting on a limited set of races has quite an effect on the results.

----

The next approach I tried is using a similar calculation but I limited the races even more.  Instead of a different of 1 between the lowest and second lowest horse, I used a difference of 2.  For example, if the lowest horse prediction is 1.20 and the second lowest is 2.90, I did not simulate a bet on the race. On the other hand, if a lowest horse prediction is 1.20 and the second lowest is 3.21, I simulated the bet.

This modification to the algorithm *should* pick races that have a horse which is much more superior than the rest.

The results are as follows:

<p align="center">
<img src="http://dplouffe.ca/static/img/offset_two.png" border="0" />
</p>

61 races met the above criteria. The horses with the lowest prediction won 33 times (54%) and came in 1st, 2nd or 3rd 51 times (84%).  Once more an improvement in the results.

As you can see, there was only 61 races that we bet on but the results are **dramatically improved**.

----

Finally, I tried one more approach.

I had was to figure out a way to understand how varied the predictions were between the horses of a race. The idea is that if there's a lot of variance between the prediction, the horse with the lowest prediction may have a better chance of winning.

To do this, I took the standard deviation of all the predictions of a race.  The standard deviation gave me a general understanding of whether the fields scored varied or not. For example, a standard deviation that was lower than 1 means the horses are very even. A race with horses which have even characteristics will be harder to predict. A standard deviation greater than 1.4 means that the favorite horse is probably quite better than the competition. If correc, the race will be easier to predict.

The results are as follows:

<p align="center">
<img src="http://dplouffe.ca/static/img/std.png" border="0" />
</p>

68 horse met had a standard deviation of 1.4 and above. The horses with the lowest prediction won 27 times (40%) and came in 1st, 2nd or 3rd 59 times (86%). Worst resuls on a win bet and similar results on the WPS bets compared to the previous results.

## So what does this mean?

Trying to educate myself on maching learning algorithm by picking challenging problems is fun. If I can bet on 61 races throughout a month and win 54%/84% of the times I will have some fun doing it. But will I make or lose money with these results?

The challenge with this algorithm is that it predicts the best horses. This is what the algorithm was designed to do and it seems to predict the best horse quite well... Unfortunately the best horse doesn't always win :(  When they do win they usually don't pay very much. North American horse racing uses a [parimutuel betting system](https://en.wikipedia.org/wiki/Parimutuel_betting). Parimutuel system means that you bet against the rest of the public, which means that the best horse will usually have lower odds and pay less money.

I chose to test the 3rd results to see how much money I would win or lose. 61 bets will cost $122.00 (A bet has a minimum wager of $2.00). To make a profit I need the 33 winning horses to return at least $123.00.

Below is the actual payout for the 33 races. You can get the raw data from [here](https://www.dropbox.com/s/gaimxdk253un3bh/Results.xlsx?dl=0):

<p>
<table>
<tr><td>Total Bets</td><td>61</td></tr>
<tr><td>Winning Bets</td><td>33</td></tr>
<tr><td>Total Wager ($2 bets)</td><td>$122.00</td></tr>
<tr><td>Total Winnings</td><td>$120.00</td></tr>
<tr><td>Difference</td><td>-$2.00</td></tr>
</table>
</p>

A loss of $2.00 is not bad, but I still loss money.

To improve the algorithm I would need to modify it to pick a different type of horse. Instead of picking the best horse, the feature set would need to define horses which are the best **but different than the public would choose**.  For example, horses that have final odds of 3 to 1 or more and has a great chance of winning. For that though, I need to find additional data points and features. Maybe another analysis on a different day!

Please feel free to email me your comments and ideas.. dominic[at]dplouffe.ca.  I hope you enjoyed my analysis.

---

## Comparison to previous month results

### January 2016

Training data from October 2016 to December 2016.

| Test Name | Num Races | Wins | WPS | Wins % | WPS % |
| --- | --- | --- | --- | --- | --- |
| SVR - Baseline | 2769 | 715 | 1569 | 26% | 57% |
| SVR - All Races | 2769 | 754 | 1663 | 27% | 60% |
| SVR - 1 Offset | 677 | 231 | 456 | 34% | 67% |
| SVR - 2 Offset | 56 | 23 | 44 | 41% | 79% |
| SVR - Std | 75 | 29 | 57 | 39% | 76% |

<p align="center">
<img src="http://dplouffe.ca/static/img/jan.png" border="0" style="width: 524px; height: 288px;" />
</p>

### February 2016

Training data from November 2016 to January 2017.

| Test Name | Num Races | Wins | WPS | Wins % | WPS % |
| --- | --- | --- | --- | --- | --- |
| SVR - Baseline | 2527 | 646 | 1482 | 26% | 59% |
| SVR - All Races | 2527 | 688 | 1557 | 27% | 62% |
| SVR - 1 Offset | 559 | 184 | 388 | 33% | 69% |
| SVR - 2 Offset | 43 | 18 | 37 | 42% | 86% |
| SVR - Std | 86 | 37 | 71 | 43% | 83% |

<p align="center">
<img src="http://dplouffe.ca/static/img/feb.png" border="0" style="width: 524px; height: 288px;" />
</p>

### March 2016

Training data from December 2016 to February 2017.

| Test Name | Num Races | Wins | WPS | Wins % | WPS % |
| --- | --- | --- | --- | --- | --- |
| SVR - Baseline | 2896 | 741 | 1666 | 26% | 58% |
| SVR - All Races | 2896 | 812 | 1820 | 28% | 63% |
| SVR - 1 Offset | 664 | 251 | 477 | 38% | 72% |
| SVR - 2 Offset | 61 | 33 | 51 | 54% | 84% |
| SVR - Std | 68 | 27 | 59 | 40% | 86% |

<p align="center">
<img src="http://dplouffe.ca/static/img/std.png" border="0" />
</p>
