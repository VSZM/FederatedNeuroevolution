from numpy import array
import json

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