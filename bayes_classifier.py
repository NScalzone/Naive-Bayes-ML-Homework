from data_processing import *
import math

TRAINING_DATA = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Naive-Bayes-ML-Homework/spambase/spambase_train.csv"
TESTING_DATA = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Naive-Bayes-ML-Homework/spambase/spambase_test.csv"

training_set = get_data(TRAINING_DATA)
test_set = get_data(TESTING_DATA)

width = len(training_set[0])

print(f"first line test data is: {test_set[0]}")
print(f"class of first line test set: {test_set[0][width-1]}")

spam_set = []
not_spam_set = []
for i in range(len(training_set)):
    if training_set[i][len(training_set[i])-1] == 1.0:
        spam_set.append(training_set[i])
    else:
        not_spam_set.append(training_set[i])


spam_size = len(spam_set)
not_spam_size = len(not_spam_set)
pclass_spam = spam_size/(len(training_set))
pclass_not_spam = not_spam_size/(len(training_set))

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

def naive_bayes_probability(x:float, mean:float, std_dev:float)->float:
    return_val = (1/(math.sqrt(2*math.pi)*std_dev))*math.exp(-(((x - mean)**2)/(2*std_dev**2)))
    return return_val

def classification(p_class:float, means:List[float], std_devs:List[float], data:List[float])->float:
    total_prob = math.log(p_class)
    for i in range(len(data)-1):
        total_prob += math.log(naive_bayes_probability(data[i], means[i],std_devs[i]) + (10**-50))
    return total_prob
    

spam_means = get_means(spam_set)
not_spam_means = get_means(not_spam_set)
spam_std_dev = get_std_dev(spam_set,spam_means)
not_spam_std_dev = get_std_dev(not_spam_set,not_spam_means)

correct_spam_results = 0
correct_not_spam_results = 0
total_spam_emails = 0

confusion_matrix = [[0,0],[0,0]]

for i in range(len(test_set)):
    spam_class = classification(pclass_spam, spam_means, spam_std_dev, test_set[i])
    not_spam_class = classification(pclass_not_spam, not_spam_means, not_spam_std_dev, test_set[i])
    
    if (test_set[i][width-1]) == 1.0:
        total_spam_emails += 1
        if spam_class > not_spam_class:
            #Correctly identified as spam
            correct_spam_results += 1
            confusion_matrix[0][0] += 1
        else:
            confusion_matrix[0][1] += 1
    else:
        if not_spam_class > spam_class:
            correct_not_spam_results += 1
            confusion_matrix[1][1] += 1
        else:
            confusion_matrix[1][0] += 1

# print(f"spam means is: {spam_means}\n\nnot spam means is: {not_spam_means}\n")
    
# print(f"spam std dev is: {spam_std_dev}\n\nnot spam std dev is: {not_spam_std_dev}\n")

# print(f"probability of spam is: {pclass_spam}\nprobability otherwise: {pclass_not_spam}")

# print(f"{correct_spam_results} spam emails identified\n{correct_not_spam_results} non-spam emails identified")
# print(f"Success rate is: {(correct_spam_results+correct_not_spam_results)/(len(test_set))}")
# print(f"Success rate on spam is: {100*(correct_spam_results/total_spam_emails)}%")
total_not_spam_emails = len(test_set)-total_spam_emails
# print(f"Success on non-spam emails is: {100*(correct_not_spam_results/(total_not_spam_emails))}%")
# print(f"Total size of test set is: {len(test_set)}")

spam_recall = round((100*(correct_spam_results/total_spam_emails)),1)
not_spam_recall = round((100*(correct_not_spam_results/(total_not_spam_emails))),1)

spam_precision = round((100*(correct_spam_results/(correct_spam_results+confusion_matrix[1][0]))),1)
spam_precision_readout = f"{correct_spam_results} out of {correct_spam_results+confusion_matrix[1][0]}, {spam_precision}%"
not_spam_precision = round((100*(correct_not_spam_results/(correct_not_spam_results+confusion_matrix[0][1]))),1)
not_spam_precision_readout = f"{correct_not_spam_results} out of {correct_not_spam_results+confusion_matrix[0][1]}, {not_spam_precision}%"

print(f"\n\t\tConfusion Matrix:\nEmail Identified as:\tSpam\tNot-Spam\tRecall\t\t\tPrecision")
print(f"Spam\t\t\t{confusion_matrix[0][0]}\t{confusion_matrix[0][1]}\t\t{confusion_matrix[0][0]} out of {total_spam_emails}, {spam_recall}%\t{spam_precision_readout}")
print(f"Not-Spam\t\t{confusion_matrix[1][0]}\t{confusion_matrix[1][1]}\t\t{confusion_matrix[1][1]} out of {total_not_spam_emails}, {not_spam_recall}%\t{not_spam_precision_readout}\n")

print(f"Accuracy: Of {len(test_set)} total emails, {correct_spam_results+correct_not_spam_results} were correctly identified")
print(f"This is {round((100*(correct_spam_results+correct_not_spam_results)/(len(test_set))),1)}% Accuracy\n\n")