import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import sys
sys.path.insert(0, '../..')  
from my_evaluation import my_evaluation
from GA import GA
import time

class my_model():
    def __init__(self):
        # Setting up our text preprocessor and initializing the classifier variable
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True)
        self.clf = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Preprocessing the input data and splitting it for training
        X = X.copy()
        X['description'].fillna('', inplace=True)  # Making sure there are no empty strings
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = self.preprocessor.fit_transform(X_train["description"])
        self.y_train = y_train

        # Objective function for our GA: Tries different hyperparameters to find the best ones
        def obj_func(X):
            alpha = 10 ** X[0]  # Regularization strength
            loss_type = 'hinge' if X[1] < 0.5 else 'log_loss'  # Choosing the loss type
            clf = SGDClassifier(loss=loss_type, alpha=alpha, max_iter=1000, tol=1e-3, random_state=42)
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=3, scoring='f1')
            return -scores.mean()

        # Running the genetic algorithm to optimize the classifier parameters
        ga = GA(function=obj_func, bounds=np.array([[-5, -1], [0, 1]]), dimension=2)
        start_time = time.time()
        best_params = ga.run()
        elapsed_time = time.time() - start_time
        print(f"GA Optimization Time: {elapsed_time:.2f} seconds")

        # Setting up the classifier with the best parameters found
        self.clf = SGDClassifier(loss='hinge' if best_params[1] < 0.5 else 'log_loss', alpha=10 ** best_params[0], max_iter=1000, tol=1e-3, random_state=42)
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, X):
        # Preprocess and predict new data using the trained classifier
        X = X.copy()
        X['description'].fillna('', inplace=True)
        XX = self.preprocessor.transform(X["description"])
        return self.clf.predict(XX)

# Loading data and setting up our model
data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
X = data[['description']]  
y = data['fraudulent']

model = my_model()
model.fit(X, y)
predictions = model.predict(X)

# Evaluating the model with our own evaluation class
evaluator = my_evaluation(predictions, y)
accuracy = evaluator.accuracy()
precision = evaluator.precision()
recall = evaluator.recall()
f1_score = evaluator.f1()

# Printing out the performance metrics
print(f"F1 Score: {f1_score}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# wangtiles@Krishnas-MacBook-Pro project % python project.py
#                                                                                            The best solution found:
#  [-4.74253305  0.47233005]

#  Objective function:
#  -0.6862437030435805
# GA Optimization Time: 513.76 seconds
# F1 Score: 0.9668409592395464
# Accuracy: 0.9937360178970918
# Precision: 0.9853957075788062
# Recall: 0.9489720588600213





