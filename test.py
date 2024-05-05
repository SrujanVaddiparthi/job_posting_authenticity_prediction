import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from project import my_model
sys.path.insert(0, '/Users/wangtiles/job_posting_authenticity_prediction/fake_job_postings.csv')
from my_evaluation import my_evaluation

def test(data):
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = my_model()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    eval = my_evaluation(predictions, y_test)
    f1 = eval.f1(target=1)
    
    accuracy = eval.accuracy()
    precision = eval.precision()
    recall = eval.recall()
    
    return f1, accuracy, precision, recall

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("/Users/wangtiles/job_posting_authenticity_prediction/fake_job_postings.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    result = test(data)
    print("F1 score: %f" % result[0])
    print("Accuracy: %f" % result[1])
    print("Precision: %f" % result[2])
    print("Recall: %f" % result[3])
    runtime = (time.time() - start) / 60.0
    print(runtime)
