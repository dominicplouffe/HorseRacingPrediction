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
        f.close()

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
                    'precition': None,
                    'finish_pos': finish_pos,
                    'odds': morning_line
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

            num_races += 1
            if horses[0]['finish_pos'] == 1:
                num_correct_pred_win += 1

            if horses[0]['finish_pos'] in [1, 2, 3]:
                num_correct_pred_wps += 1

        print('Number of races predicted => %s' % num_races)
        print('Number of correct win predictions = %s' % num_correct_pred_win)
        print('Number of correct WPS predictions = %s' % num_correct_pred_wps)


if __name__ == '__main__':

    trn = Model()
    #trn.train()

    trn.predict()
