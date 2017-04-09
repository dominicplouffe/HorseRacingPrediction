import csv
import pickle
import logging
import numpy as np
from sklearn.svm import SVR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(object):

    def __init__(self):
        pass

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
            X.append(data)

        return X, y

    def train(self):

        clf = SVR(C=1.0, epsilon=0.1, cache_size=1000)
        X, y, = self._get_data('training-2016-12-01-2017-02-28.csv')

        # Fit the model
        clf.fit(X, y)

        # Pickle the model so we can save and reuse it
        s = pickle.dumps(clf)

        # Save the model to a file
        f = open('finish_pos.model', 'wb')
        f.write(s)
        f.close()

    def predict(self):
        f = open('finish_pos.model', 'rb')
        clf = pickle.loads(f.read())
        f.close()

        validation_data = csv.reader(
            open('data/validation-2017-03-01-2017-03-31.csv', 'rb')
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
                float(_ if len(str(_)) > 0 else 0)
                for _ in row[5:-1]
            ])
            data = data.reshape(1, -1)

            morning_line = row[0].split('-')
            morning_line = float(morning_line[0]) / float(morning_line[1])
            races[race_id].append(
                {
                    'entry_id': row[0],
                    'data': data,
                    'prediction': None,
                    'finish_pos': finish_pos,
                    'odds': morning_line
                }
            )

        for race_id, horses in races.iteritems():
            for horse in horses:
                horse['prediction'] = clf.predict(
                    horse['data']
                )

        tests = [
            'SVR - Baseline',
            'SVR - All Races',
            'SVR - 1 Offset',
            'SVR - 2 Offset',
            'SVR - Std',
            'SVR - 2 Offset + Std'
        ]

        print('Test Name\tNum Races\tWins\tWPS')

        for test in tests:
            num_races = 0
            num_correct_pred_win = 0
            num_correct_pred_wps = 0
            for race_id, horses in races.iteritems():

                if len(horses) < 2:
                    continue
                diff = horses[1]['prediction'] - horses[0]['prediction']
                std = np.std([x['prediction'] for x in horses])

                if test == 'SVR - Baseline':
                    horses.sort(key=lambda x: x['odds'])
                else:
                    horses.sort(key=lambda x: x['prediction'])

                if test == 'SVR - 1 Offset' and diff < 1.0:
                    continue
                if test == 'SVR - 2 Offset' and diff < 2.0:
                    continue
                if test == 'SVR - Std' and std < 1.4:
                    continue
                if test == 'SVR - 2 Offset + Std' and (
                    std < 1.4 or diff < 2.0
                ):
                    continue

                num_races += 1
                if horses[0]['finish_pos'] == 1:
                    num_correct_pred_win += 1

                if horses[0]['finish_pos'] in [1, 2, 3]:
                    num_correct_pred_wps += 1

            print('%s\t%s\t%s\t%s' % (
                test,
                num_races,
                num_correct_pred_win,
                num_correct_pred_wps
            ))


if __name__ == '__main__':

    trn = Model()
    trn.train()
    trn.predict()
