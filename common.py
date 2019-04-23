from numpy import array
import json
import keras
from keras import metrics
import keras.backend as K
try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except:
    from tqdm import tqdm
from typing import List
import numpy as np
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import pickle
from functools import wraps
from time import time
import datetime
import matplotlib
import matplotlib.pyplot

log = logging.getLogger(__name__)


# Mapping channel names, to indexes
channel_map = {
    'FP1': 0,
    'FP2': 1,
    'F7': 2,
    'F8': 3,
    'AF1': 4,
    'AF2': 5,
    'FZ': 6,
    'F4': 7,
    'F3': 8,
    'FC6': 9,
    'FC5': 10,
    'FC2': 11,
    'FC1': 12,
    'T8': 13,
    'T7': 14,
    'CZ': 15,
    'C3': 16,
    'C4': 17,
    'CP5': 18,
    'CP6': 19,
    'CP1': 20,
    'CP2': 21,
    'P3': 22,
    'P4': 23,
    'PZ': 24,
    'P8': 25,
    'P7': 26,
    'PO2': 27,
    'PO1': 28,
    'O2': 29,
    'O1': 30,
    'X': 31,
    'AF7': 32,
    'AF8': 33,
    'F5': 34,
    'F6': 35,
    'FT7': 36,
    'FT8': 37,
    'FPZ': 38,
    'FC4': 39,
    'FC3': 40,
    'C6': 41,
    'C5': 42,
    'F2': 43,
    'F1': 44,
    'TP8': 45,
    'TP7': 46,
    'AFZ': 47,
    'CP3': 48,
    'CP4': 49,
    'P5': 50,
    'P6': 51,
    'C1': 52,
    'C2': 53,
    'PO7': 54,
    'PO8': 55,
    'FCZ': 56,
    'POZ': 57,
    'OZ': 58,
    'P2': 59,
    'P1': 60,
    'CPZ': 61,
    'nd': 62,
    'Y': 63
}

def timed_method(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        log.debug('%r  %2.2f ms', f.__name__, (te - ts) * 1000)
        return result
    return wrap

class Trial(object):


    def __init__(self, subject_id: str, subject_class: int, trial_number: int, trial_type: int, eeg: array):
        """
            subject_id is like: co2a0000364
            subject_class: 1 for alcoholic and 0 for control 
            trial_type: 0 for S1 obj, 1 for S2 match, 2 for S2 nomatch
            eeg: a 2d array of shape (64, 256) containing 256 microvolt measurement of 64 electrodes in floats
        """
        self.subject_id = subject_id
        self.subject_class = subject_class
        self.trial_number = trial_number
        self.trial_type = trial_type
        self.eeg = eeg

    def __hash__(self):
        return self.subject_id.__hash__()
    
    def __eq__(self, other):
        if other == None:
            return False

        return self.subject_id == other.subject_id and self.trial_number == other.trial_number

    def to_dict(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self.to_dict(), default=str, indent=4, separators=(',', ': '), sort_keys=False)

    def __repr__(self):
        return self.__str__()
    

def plot_learning(best_of_each_generation, num_generations):
    matplotlib.pyplot.plot(best_of_each_generation, linewidth=5, color="black")
    matplotlib.pyplot.xlabel("Iteration", fontsize=20)
    matplotlib.pyplot.ylabel("Accuracy", fontsize=20)
    matplotlib.pyplot.xticks(np.arange(0, num_generations + 1, num_generations / 5), fontsize=15)
    matplotlib.pyplot.yticks(np.arange(0, num_generations + 1, num_generations / 5), fontsize=15)

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return K.log(K.clip(x, eps, 100000000))


def ts():
    return str(datetime.datetime.now()).split(' ')[0]

def read_trials(eeg_file: str) -> List[Trial]:
    errors, zeros = 0, 0
    
    with open(eeg_file, 'r') as f:
        file_content = f.read()
        
        subject_id = file_content[2:13]
        if 'a' == subject_id[3]:
            subject_class = 1
        elif 'c' == subject_id[3]:
            subject_class = 0
        else:
            raise ValueError('Invalid subject class: ' + subject_id)
        
        trials = []
        trials_str = list(filter(lambda string: len(string) > 0, re.split(r'^# co\w{9}.rd', file_content, flags=re.MULTILINE)))

        
        for trial in trials_str:
            try:
                lines = list(filter(lambda line: len(line) > 0, map(lambda line: line.strip(), trial.split('\n'))))
                trial_type_str, trial_number = tuple(lines[2].split(', trial'))
                trial_number = int(trial_number)

                if 'err' in trial_type_str:
                    log.warn('Skipping trial |%d| from file |%s| due to error type', trial_number, eeg_file)
                    errors = errors + 1
                    continue
                elif trial_type_str.startswith('# S1 obj'):
                    trial_type = 0
                elif trial_type_str.startswith('# S2 match'):
                    trial_type = 1
                elif trial_type_str.startswith('# S2 nomatch'):
                    trial_type = 2
                else:
                    raise ValueError('Invalid trial_type_str: ' + trial_type_str)

                lines = lines[4:]

                measurements = [(channel_map[line.split()[1]], int(line.split()[2]), float(line.split()[3])) for line in lines if len(line) > 0 and line[0] != '#']
                measurements = np.array(measurements)
                eeg = measurements[:, 2].reshape((64,256))
                #eeg = np.array([np.mean(eeg_channel.reshape(-1, 4), axis=1) for eeg_channel in eeg[::1]])
                #eeg = (eeg - np.min(eeg))/np.ptp(eeg)
                

                if np.count_nonzero(eeg) == 0:
                    log.warn('Skipping trial |%d| from file |%s| due to only 0 values', trial_number, eeg_file)
                    zeros = zeros + 1
                    continue
                    
                trials.append(Trial(subject_id, subject_class, trial_number, trial_type, eeg))
            except:
                log.exception('Error in file: |%s|', eeg_file)
                log.error('Error for: |%s|', trial)
                log.error('Error for: |%s|', lines[2])
                raise
                
            
        return trials, errors, zeros


def __read_all_trials():
    errors = 0
    zeros = 0

    all_trials = []

    for file in tqdm(os.listdir('eeg_full')):
        current_trials, error, zero =  read_trials('eeg_full/' + file)
        all_trials = all_trials + current_trials
        errors = errors + error
        zeros = zeros + zero
        
    log.info('Good trials: %d, Error trials: %d, Zeros trials: %d', len(all_trials), errors, zeros)

    return all_trials

def df_to_ML_data(df):
    X = df['eeg'].values
    # Zero Mean Unit Variance
    y = df['subject_class'].values

    # keras required format
    X = np.rollaxis(np.dstack(X), -1)
    X = X.reshape(X.shape[0], 64, 256, 1)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = keras.utils.to_categorical(y, 2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    y_train_argmax = np.argmax(y_train, axis = 1)
    y_test_argmax = np.argmax(y_test, axis = 1)


    return X_train, X_test, y_train, y_test, y_train_argmax, y_test_argmax

def load_df():
    if os.path.isfile('df.pkl'):
        with open('df.pkl', 'rb') as f:
            df = pickle.load(f)
    else:
        all_trials = __read_all_trials()

        nans = [trial for trial in all_trials if np.isnan(trial.eeg).any()]

        log.info('Nans: %s', nans)

        df = pd.DataFrame.from_records([trial.to_dict() for trial in all_trials])
        del all_trials
        with open('df.pkl', 'wb') as f:
            pickle.dump(df, f)

    # balancing sample sizes per class
    df = df.groupby('subject_class')
    df = df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True))
    df = df.sample(frac=1).reset_index(drop=True)


    return df