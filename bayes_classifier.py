from data_processing import *
import math

TRAINING_DATA = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Naive-Bayes-ML-Homework/spambase/spambase_train.csv"
TESTING_DATA = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Naive-Bayes-ML-Homework/spambase/spambase_test.csv"

training_set = get_data(TRAINING_DATA)

width = len(training_set[0])

print(f"first line training data is: {training_set[0]}")

spam_set = []
not_spam_set = []
for i in range(len(training_set)):
    if training_set[i][len(training_set[i])-1] == 1.0:
        spam_set.append(training_set[i])
    else:
        not_spam_set.append(training_set[i])


spam_size = len(spam_set)
not_spam_size = len(not_spam_set)

def get_means(data_set)->list[float]:
    to_return = []
    for i in range(width-1):
        to_return.append(0)
        
    for i in range(len(data_set)):
        for j in range(len(data_set[i]) - 1):
            to_return[j] += data_set[i][j]

    for i in range(len(to_return)):
        to_return[i] = to_return[i]/(len(data_set))

    return to_return

def get_std_dev(data_set, means_set)->List[float]:
    std_dev_set = []
    for i in range(width-1):
        std_dev_set.append(0)
    
    for i in range(len(data_set)):
        for j in range(len(data_set[i]) - 1):
            std_dev_set[j] += (data_set[i][j] - means_set[j])**2

    for i in range(len(std_dev_set)):
        std_dev_set[i] = math.sqrt(std_dev_set[i]/(width-1))
        if std_dev_set[i] == 0:
            std_dev_set[i] = 0.0001 
    return std_dev_set


spam_means = get_means(spam_set)
not_spam_means = get_means(not_spam_set)
spam_std_dev = get_std_dev(spam_set,spam_means)
not_spam_std_dev = get_std_dev(not_spam_set,not_spam_means)

print(f"spam means is: {spam_means}\n\nnot spam means is: {not_spam_means}\n")
    
print(f"spam std dev is: {spam_std_dev}\n\nnot spam std dev is: {not_spam_std_dev}\n")
