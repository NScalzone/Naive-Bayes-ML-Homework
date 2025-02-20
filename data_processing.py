import random
import csv
from typing import List

TRAINING_DATA = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Naive-Bayes-ML-Homework/spambase/spambase_train.csv"
TESTING_DATA = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Naive-Bayes-ML-Homework/spambase/spambase_test.csv"

def generate_weights(units:int, unit_length:int)->List[float]:
    model_weights = []
    for i in range(units):
        temp = []
        for j in range(unit_length):
            temp.append(round((random.uniform(-0.5,0.5)),1))
        model_weights.append(temp)
    return model_weights

def get_data(path:str)->List[float]:    
    data = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            data.append(row)
    return data
