# import pandas as pd
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# import time
# import sys
# sys.path.insert(0,'../..')
# from my_evaluation import my_evaluation
# from GA import GA

# class my_model():

#     def obj_func(self, predictions, actuals, pred_proba=None):
#         # One objectives: higher f1 score
#         eval = my_evaluation(predictions, actuals, pred_proba)
#         return [eval.f1()]

#     def fit(self, X, y):
#         # do not exceed 29 mins
#         self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
#         XX = self.preprocessor.fit_transform(X["description"])
#         XX = pd.DataFrame(XX.toarray())
#         # Use your GA to optimize hyperparemters for a model to get "best_parameters"
#         self.clf = SGDClassifier(best_parameters)
#         self.clf.fit(XX,y)
#         return

#     def predict(self, X):
#         # remember to apply the same preprocessing in fit() on test data before making predictions
#         XX = self.preprocessor.transform(X["description"])
#         predictions = self.clf.predict(XX)
#         return predictions
# fdaffdafasfsa
# import pandas as pd
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import SGDClassifier
# import numpy as np
# from geneticalgorithm import geneticalgorithm as ga
# from my_evaluation import my_evaluation
# import time

# class my_model():
#     def __init__(self):
#         self.pipeline = None
#         self.best_params = None
#         self.vectorizer = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
    
#     def obj_func(self, variables):
#         # Mapping GA variables to model hyperparameters
#         # Please ensure these are in the correct range for your GA setup
#         params = {
#             'loss': 'hinge' if variables[0] <= 0.5 else 'log',
#             'penalty': 'l2' if variables[1] <= 0.5 else 'l1',
#             'alpha': max(min(variables[2], 0.1), 0.0001),  # Ensuring alpha is within a reasonable range
#             'max_iter': int(variables[3]),
#             'random_state': 42
#         }
#         clf = SGDClassifier(**params)
#         pipeline = make_pipeline(self.vectorizer, clf)
        
#         # Cross-validation and scoring
#         scores = cross_val_score(pipeline, self.X['description'], self.y, scoring='f1', cv=5)
#         return -scores.mean()  # GA minimizes by default, so we negate the score
    
#     def fit(self, X, y):
#         self.X = X
#         self.y = y
        
#         # Define the boundaries for the GA
#         varbound = np.array([[0, 1], [0, 1], [0.0001, 0.1], [100, 1000]])

#         # GA setup
#         algorithm_param = {
#             'max_num_iteration': 500,
#             'population_size':100,
#             'mutation_probability':0.1,
#             'elit_ratio': 0.01,
#             'crossover_probability': 0.5,
#             'parents_portion': 0.3,
#             'crossover_type':'uniform',
#             'max_iteration_without_improv':None
#         }

#         model = ga(function=self.obj_func,
#                    dimension=4,
#                    variable_type='real',
#                    variable_boundaries=varbound,
#                    algorithm_parameters=algorithm_param)

#         # Running GA
#         start_time = time.time()
#         model.run()
#         run_time = time.time() - start_time
#         print(f"GA run time: {run_time:.2f} seconds")

#         # Fetch the best parameters and retrain the model on the entire dataset
#         self.best_params = model.output_dict['variable']
#         best_clf_params = {
#             'loss': 'hinge' if self.best_params[0] <= 0.5 else 'log',
#             'penalty': 'l2' if self.best_params[1] <= 0.5 else 'l1',
#             'alpha': max(min(self.best_params[2], 0.1), 0.0001),
#             'max_iter': int(self.best_params[3]),
#             'random_state': 42
#         }
#         self.pipeline = make_pipeline(
#             TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False),
#             SGDClassifier(**best_clf_params)
#         )
#         self.pipeline.fit(self.X['description'], self.y)
    
#     def predict(self, X):
#         return self.pipeline.predict(X['description'])

# # Assuming you have a function to load your dataset
# # Replace 'path_to_dataset.csv' with the actual path to your dataset
# dataset_path = '/Users/wangtiles/DSCI-633/assignments/data/job_train.csv'
# data = pd.read_csv(dataset_path)
# X = data.drop('fraudulent', axis=1)
# y = data['fraudulent']

# model = my_model()
# model.fit(X, y)
# predictions = model.predict(X)  # Example prediction on training set

# # Example evaluation using your my_evaluation class
# eval = my_evaluation(predictions, y)
# print(f"F1 Score: {eval.f1()}")
# ____________
# import pandas as pd
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.exceptions import NotFittedError
# import numpy as np
# from geneticalgorithm import geneticalgorithm as ga
# from my_evaluation import my_evaluation
# import time

# class my_model():
#     def __init__(self):
#         self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
#         self.clf = None
#         self.X = None
#         self.y = None

#     def obj_func(self, variables):
#         # Map variables to model hyperparameters
#         mapped_params = {
#             'loss': 'hinge' if variables[0] < 0.5 else 'log_loss',
#             'penalty': 'l2' if variables[1] < 0.5 else 'l1',
#             'alpha': max(min(variables[2], 0.1), 0.0001),
#             'max_iter': int(variables[3]),
#             'random_state': 42
#         }
#         clf = SGDClassifier(**mapped_params)

#         pipeline = Pipeline([
#             ('vectorizer', self.preprocessor),
#             ('classifier', clf)
#         ])
        
#         scores = cross_val_score(pipeline, self.X_train['description'].fillna(''), self.y_train, scoring='f1', cv=5)
#         return -np.mean(scores)  # Minimize the negative F1 score

#     def fit(self, X, y):
#         self.X = X.copy()
#         self.y = y
#         self.X['description'].fillna('', inplace=True)  # Replace NaN with empty string
        
#         # Split the data into training and testing sets
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

#         # Set the bounds for the hyperparameters to optimize
#         varbound = np.array([[0, 1], [0, 1], [0.0001, 0.1], [1000, 2000]])

#         # Define genetic algorithm
#         model = ga(function=self.obj_func,
#                    dimension=4,
#                    variable_type='real',
#                    variable_boundaries=varbound)

#         start_time = time.time()
#         model.run()
#         print(f"Genetic Algorithm took {time.time() - start_time:.2f} seconds")

#         # Get the best parameters
#         best_params = model.output_dict['variable']
#         self.clf = SGDClassifier(loss='hinge' if best_params[0] < 0.5 else 'log_loss',
#                                  penalty='l2' if best_params[1] < 0.5 else 'l1',
#                                  alpha=best_params[2],
#                                  max_iter=int(best_params[3]),
#                                  random_state=42)
        
#         # Fit the vectorizer and the classifier on the full training data
#         self.clf.fit(self.preprocessor.fit_transform(self.X_train['description']), self.y_train)

#     def predict(self, X):
#         # Check if the model and vectorizer have been fitted
#         try:
#             check_is_fitted(self.clf)
#             check_is_fitted(self.preprocessor)
#         except NotFittedError:
#             raise NotFittedError("This my_model instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

#         # Preprocess and predict
#         XX = self.preprocessor.transform(X['description'].fillna(''))
#         predictions = self.clf.predict(XX)
#         return predictions

# # Assuming 'path_to_your_data.csv' is the path to the dataset
# # Make sure to change this path to where your actual dataset is located
# data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
# X = data.drop(columns=['fraudulent'])
# y = data['fraudulent']

# # Create an instance of the model and fit it
# model = my_model()
# model.fit(X, y)

# # Make predictions on the testing data
# predictions = model.predict(X)

# # Evaluate the model using my_evaluation
# evaluator = my_evaluation(predictions, y)
# print(f"F1 Score on Testing Data: {evaluator.f1()}")



# import pandas as pd
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import cross_val_score
# from sklearn.exceptions import NotFittedError
# import numpy as np
# from geneticalgorithm import geneticalgorithm as ga
# from my_evaluation import my_evaluation
# import time

# class my_model():
#     def __init__(self):
#         # Use a simple tokenizer for TfidfVectorizer to improve preprocessing speed
#         self.preprocessor = TfidfVectorizer(stop_words='english', tokenizer=lambda x: x.split(), use_idf=False)
#         self.clf = None
#         self.X = None
#         self.y = None

#     def obj_func(self, variables):
#         # Map variables to model hyperparameters
#         mapped_params = {
#             'loss': 'hinge' if variables[0] < 0.5 else 'log_loss',
#             'penalty': 'l2' if variables[1] < 0.5 else 'l1',
#             'alpha': max(min(variables[2], 0.1), 0.0001),
#             'max_iter': int(variables[3]),
#             'random_state': 42
#         }
#         clf = SGDClassifier(**mapped_params)
        
#         # Preprocess data using TfidfVectorizer
#         X_processed = self.preprocessor.fit_transform(self.X['description'].fillna(''))

#         # Calculate F1 score using cross-validation
#         scores = cross_val_score(clf, X_processed, self.y, scoring='f1', cv=5)
#         return -np.mean(scores)  # Minimize the negative F1 score

#     def fit(self, X, y):
#         self.X = X.copy()
#         self.y = y
#         self.X['description'].fillna('', inplace=True)  # Replace NaN with empty string

#         # Set the bounds for the hyperparameters to optimize
#         varbound = np.array([[0, 1], [0, 1], [0.0001, 0.1], [1000, 2000]])

#         # Define genetic algorithm
#         model = ga(function=self.obj_func,
#                    dimension=4,
#                    variable_type='real',
#                    variable_boundaries=varbound)

#         start_time = time.time()
#         model.run()
#         print(f"Genetic Algorithm took {time.time() - start_time:.2f} seconds")

#         # Get the best parameters
#         best_params = model.output_dict['variable']
#         self.clf = SGDClassifier(loss='hinge' if best_params[0] < 0.5 else 'log_loss',
#                                  penalty='l2' if best_params[1] < 0.5 else 'l1',
#                                  alpha=best_params[2],
#                                  max_iter=int(best_params[3]),
#                                  random_state=42)
        
#         # Preprocess data using TfidfVectorizer
#         X_processed = self.preprocessor.fit_transform(self.X['description'].fillna(''))

#         # Fit the classifier on the preprocessed data
#         self.clf.fit(X_processed, self.y)

#     def predict(self, X):
#         # Check if the model has been fitted
#         if self.clf is None:
#             raise NotFittedError("This my_model instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

#         # Preprocess input data using TfidfVectorizer
#         X_processed = self.preprocessor.transform(X['description'].fillna(''))

#         # Predict using the fitted classifier
#         predictions = self.clf.predict(X_processed)
#         return predictions

# # Assuming 'path_to_your_data.csv' is the path to the dataset
# # Make sure to change this path to where your actual dataset is located
# data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
# X = data.drop(columns=['fraudulent'])
# y = data['fraudulent']

# # Create an instance of the model and fit it
# model = my_model()
# model.fit(X, y)

# # Make predictions (Here we use the same data for simplicity, replace with actual test data)
# predictions = model.predict(X)

# # Evaluate the model using my_evaluation
# evaluator = my_evaluation(predictions, y)
# print(f"F1 Score: {evaluator.f1()}")

# # THE 1 WORKED
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from geneticalgorithm import geneticalgorithm as ga
from my_evaluation import my_evaluation
import time

class my_model():
    def __init__(self):
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False, max_df=0.9, min_df=0.01, max_features=5000)
        self.clf = None
        self.X = None
        self.y = None

    def obj_func(self, variables):
        # Map variables to model hyperparameters
        alpha = max(min(variables[0], 1), 0.0001)  # Ensure alpha is within range [0.0001, 1]
        clf = MultinomialNB(alpha=alpha)

        # Preprocess text data
        X_processed = self.preprocessor.fit_transform(self.X['description'].fillna(''))

        # Cross-validate the model
        scores = cross_val_score(clf, X_processed, self.y, scoring='f1', cv=3)
        return -scores.mean()  # Minimize the negative F1 score

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y
        self.X['description'].fillna('', inplace=True)  # Replace NaN with empty string

        # Set the bounds for the hyperparameters to optimize
        varbound = np.array([[0.0001, 1]])

        # Define genetic algorithm
        model = ga(function=self.obj_func,
                   dimension=1,
                   variable_type='real',
                   variable_boundaries=varbound)

        start_time = time.time()
        model.run()
        print(f"Genetic Algorithm took {time.time() - start_time:.2f} seconds")

        # Get the best parameters
        best_params = model.output_dict['variable']
        self.clf = MultinomialNB(alpha=best_params[0])
        
        # Fit the classifier on the preprocessed data
        X_processed = self.preprocessor.fit_transform(self.X['description'].fillna(''))
        self.clf.fit(X_processed, self.y)

    def predict(self, X):
        # Preprocess and predict
        XX = self.preprocessor.transform(X['description'].fillna(''))
        predictions = self.clf.predict(XX)
        return predictions

# Load dataset
data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
X = data.drop(columns=['fraudulent'])
y = data['fraudulent']

# Create an instance of the model and fit it
model = my_model()
model.fit(X, y)

# Make predictions (Here we use the same data for simplicity, replace with actual test data)
predictions = model.predict(X)

# Evaluate the model using my_evaluation
evaluator = my_evaluation(predictions, y)
print(f"F1 Score: {evaluator.f1()}")
# The best solution found:
#  [0.01184651]

#  Objective function:
#  -0.403101105609506
# Genetic Algorithm took 2375.91 seconds
# F1 Score: 0.7903490629582756


# 2 WORKED
# import pandas as pd
# import numpy as np
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import cross_val_score
# from geneticalgorithm import geneticalgorithm as ga
# # from GA import GA as ga
# from my_evaluation import my_evaluation
# import time

# class my_model():
#     def __init__(self):
#         # Initialize the TfidfVectorizer with specified parameters
#         self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False, max_df=0.9, min_df=0.01, max_features=5000)
#         self.clf = None
#         self.X_processed = None
#         self.y = None

#     def fit(self, X, y):
#         # Preprocess the text data once
#         self.X_processed = self.preprocessor.fit_transform(X['description'].fillna(''))
#         self.y = y

#         # Objective function for the genetic algorithm
#         def obj_func(variables):
#             alpha = max(min(variables[0], 1), 0.0001)  # Ensure alpha is within the specified range
#             clf = MultinomialNB(alpha=alpha)
#             scores = cross_val_score(clf, self.X_processed, self.y, scoring='f1', cv=3)
#             return -scores.mean()  # Minimize the negative F1 score

#         # Define the bounds for the genetic algorithm
#         varbound = np.array([[0.0001, 1]])

#         # Initialize the genetic algorithm with specified parameters
#         model = ga(function=obj_func,
#                    dimension=1,
#                    variable_type='real',
#                    variable_boundaries=varbound,
#                    algorithm_parameters={
#                        'max_num_iteration': 100,
#                        'population_size': 50,
#                        'mutation_probability': 0.1,
#                        'elit_ratio': 0.01,
#                        'crossover_probability': 0.7,
#                        'parents_portion': 0.3,
#                        'crossover_type':'uniform',
#                        'max_iteration_without_improv': 100
#                    })

#         # Run the genetic algorithm
#         start_time = time.time()
#         model.run()
#         elapsed_time = time.time() - start_time

#         # Print results and update the classifier with the best found parameters
#         print(f"Genetic Algorithm took {elapsed_time:.2f} seconds")
#         best_params = model.output_dict['variable']
#         print(f"The best solution found:\n {best_params}\n")
#         print(f"Objective function:\n {model.output_dict['function']}\n")
#         self.clf = MultinomialNB(alpha=best_params[0])
#         self.clf.fit(self.X_processed, self.y)

#     def predict(self, X):
#         # Preprocess and predict
#         X_processed = self.preprocessor.transform(X['description'].fillna(''))
#         return self.clf.predict(X_processed)

# # Load dataset
# data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
# X = data.drop(columns=['fraudulent'])
# y = data['fraudulent']

# # Create an instance of the model and fit it
# model = my_model()
# model.fit(X, y)

# # Make predictions (Here we use the same data for simplicity; replace with actual test data)
# predictions = model.predict(X)

# # Evaluate the model using your custom evaluation function
# evaluator = my_evaluation(predictions, y)
# print(f"F1 Score: {evaluator.f1()}")
# # The best solution found:
# #  [0.0123486]

# #  Objective function:
# #  -0.403101105609506
# # Genetic Algorithm took 51.91 seconds
# # The best solution found:
# #  [0.0123486]

# # Objective function:
# #  -0.403101105609506

# # F1 Score: 0.7903490629582756


# project_hint.py
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# import numpy as np
# import sys
# sys.path.insert(0, '../..')  # Adjust this path as needed
# from my_evaluation import my_evaluation
# from GA import GA
# import time

# class my_model():
#     def __init__(self):
#         self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True)
#         self.clf = None
#         self.X_train = None
#         self.y_train = None

#     def fit(self, X, y):
#         # Create a copy of X to avoid SettingWithCopyWarning when filling NaNs
#         X = X.copy()
#         X['description'].fillna('', inplace=True)
#         X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
#         self.X_train = self.preprocessor.fit_transform(X_train["description"])
#         self.y_train = y_train

#         def obj_func(X):
#             alpha = 10 ** X[0]
#             loss_type = 'hinge' if X[1] < 0.5 else 'log_loss'
#             clf = SGDClassifier(loss=loss_type, alpha=alpha, max_iter=1000, tol=1e-3, random_state=42)
#             scores = cross_val_score(clf, self.X_train, self.y_train, cv=3, scoring='f1')
#             return -scores.mean()

#         ga = GA(function=obj_func, bounds=np.array([[-5, -1], [0, 1]]), dimension=2)
#         start_time = time.time()
#         best_params = ga.run()
#         elapsed_time = time.time() - start_time
#         print(f"GA Optimization Time: {elapsed_time:.2f} seconds")

#         self.clf = SGDClassifier(loss='hinge' if best_params[1] < 0.5 else 'log_loss', alpha=10 ** best_params[0], max_iter=1000, tol=1e-3, random_state=42)
#         self.clf.fit(self.X_train, self.y_train)

#     def predict(self, X):
#         # Create a copy to avoid SettingWithCopyWarning
#         X = X.copy()
#         X['description'].fillna('', inplace=True)
#         XX = self.preprocessor.transform(X["description"])
#         return self.clf.predict(XX)

# # Example usage
# data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
# X = data[['description']]  # Adjust column selection as necessary
# y = data['fraudulent']

# model = my_model()
# model.fit(X, y)
# predictions = model.predict(X)

# # Evaluate the model
# evaluator = my_evaluation(predictions, y)
# print(f"F1 Score: {evaluator.f1()}")
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# import numpy as np
# import sys
# sys.path.insert(0, '../..')  # Adjust this path as needed
# from my_evaluation import my_evaluation
# from GA import GA
# import time

# class my_model():
#     def __init__(self):
#         self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True)
#         self.clf = None
#         self.X_train = None
#         self.y_train = None

#     def fit(self, X, y):
#         # Create a copy of X to avoid SettingWithCopyWarning when filling NaNs
#         X = X.copy()
#         X['description'].fillna('', inplace=True)
#         X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
#         self.X_train = self.preprocessor.fit_transform(X_train["description"])
#         self.y_train = y_train

#         def obj_func(X):
#             alpha = 10 ** X[0]
#             loss_type = 'hinge' if X[1] < 0.5 else 'log_loss'
#             clf = SGDClassifier(loss=loss_type, alpha=alpha, max_iter=1000, tol=1e-3, random_state=42)
#             scores = cross_val_score(clf, self.X_train, self.y_train, cv=3, scoring='f1')
#             return -scores.mean()

#         ga = GA(function=obj_func, bounds=np.array([[-5, -1], [0, 1]]), dimension=2)
#         start_time = time.time()
#         best_params = ga.run()
#         elapsed_time = time.time() - start_time
#         print(f"GA Optimization Time: {elapsed_time:.2f} seconds")

#         self.clf = SGDClassifier(loss='hinge' if best_params[1] < 0.5 else 'log_loss', alpha=10 ** best_params[0], max_iter=1000, tol=1e-3, random_state=42)
#         self.clf.fit(self.X_train, self.y_train)

#     def predict(self, X):
#         # Create a copy to avoid SettingWithCopyWarning
#         X = X.copy()
#         X['description'].fillna('', inplace=True)
#         XX = self.preprocessor.transform(X["description"])
#         return self.clf.predict(XX)

# # Example usage
# data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
# X = data[['description']]  # Adjust column selection as necessary
# y = data['fraudulent']

# model = my_model()
# model.fit(X, y)
# predictions = model.predict(X)

# # Evaluate the model
# evaluator = my_evaluation(predictions, y)
# print(f"F1 Score: {evaluator.f1()}")
# wangtiles@Krishnas-MacBook-Pro project % python project_hint.py
#                                                                                            The best solution found:
#  [-4.98836644  0.27225443]

#  Objective function:
#  -0.6860884840377626
# GA Optimization Time: 550.37 seconds
# F1 Score: 0.9686206672160428


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# import numpy as np
# import sys
# sys.path.insert(0, '/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')  # Adjust this path as needed
# from my_evaluation import my_evaluation
# from GA import GA
# import time

# class my_model():
#     def __init__(self):
#         self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True)
#         self.clf = None
#         self.X_train = None
#         self.y_train = None

#     def fit(self, X, y):
#         # Create a copy of X to avoid SettingWithCopyWarning when filling NaNs
#         X = X.copy()
#         X['description'].fillna('', inplace=True)
#         X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
#         self.X_train = self.preprocessor.fit_transform(X_train["description"])
#         self.y_train = y_train

#         def obj_func(X):
#             alpha = 10 ** X[0]
#             loss_type = 'hinge' if X[1] < 0.5 else 'log_loss'
#             clf = SGDClassifier(loss=loss_type, alpha=alpha, max_iter=1000, tol=1e-3, random_state=42)
#             scores = cross_val_score(clf, self.X_train, self.y_train, cv=3, scoring='f1')
#             return -scores.mean()

#         ga = GA(function=obj_func, bounds=np.array([[-5, -1], [0, 1]]), dimension=2)
#         start_time = time.time()
#         best_params = ga.run()
#         elapsed_time = time.time() - start_time
#         print(f"GA Optimization Time: {elapsed_time:.2f} seconds")

#         self.clf = SGDClassifier(loss='hinge' if best_params[1] < 0.5 else 'log_loss', alpha=10 ** best_params[0], max_iter=1000, tol=1e-3, random_state=42)
#         self.clf.fit(self.X_train, self.y_train)

#     def predict(self, X):
#         # Create a copy to avoid SettingWithCopyWarning
#         X = X.copy()
#         X['description'].fillna('', inplace=True)
#         XX = self.preprocessor.transform(X["description"])
#         return self.clf.predict(XX)

# # Example usage
# data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
# X = data[['description']]  # Adjust column selection as necessary
# y = data['fraudulent']

# model = my_model()
# model.fit(X, y)
# predictions = model.predict(X)

# # Evaluate the model
# evaluator = my_evaluation(predictions, y)
# print(f"F1 Score: {evaluator.f1()}")



# project_hint.py

# project_hint.py

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# import numpy as np
# import sys
# sys.path.insert(0, '../..')  # Adjust this path as needed
# from my_evaluation import my_evaluation
# from GA import GA
# import time

# class my_model():
#     def __init__(self):
#         self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True)
#         self.clf = None
#         self.X_train = None
#         self.y_train = None

#     def fit(self, X, y):
#         # Create a copy of X to avoid SettingWithCopyWarning when filling NaNs
#         X = X.copy()
#         X['description'].fillna('', inplace=True)
#         X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
#         self.X_train = self.preprocessor.fit_transform(X_train["description"])
#         self.y_train = y_train

#         def obj_func(X):
#             n_estimators = int(10 ** X[0])
#             max_depth = int(10 ** X[1])
#             clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
#             scores = cross_val_score(clf, self.X_train, self.y_train, cv=3, scoring='f1')
#             return -scores.mean()

#         ga = GA(function=obj_func, bounds=np.array([[1, 4], [1, 4]]), dimension=2, timeout=60)  # Increased timeout to 60 seconds
#         start_time = time.time()
#         best_params = ga.run()
#         elapsed_time = time.time() - start_time
#         print(f"GA Optimization Time: {elapsed_time:.2f} seconds")

#         self.clf = RandomForestClassifier(n_estimators=int(10 ** best_params[0]), max_depth=int(10 ** best_params[1]), random_state=42)
#         self.clf.fit(self.X_train, self.y_train)

#     def predict(self, X):
#         # Create a copy to avoid SettingWithCopyWarning
#         X = X.copy()
#         X['description'].fillna('', inplace=True)
#         XX = self.preprocessor.transform(X["description"])
#         return self.clf.predict(XX)

# # Example usage
# data = pd.read_csv('/Users/wangtiles/DSCI-633/assignments/data/job_train.csv')
# X = data[['description']]  # Adjust column selection as necessary
# y = data['fraudulent']

# model = my_model()
# model.fit(X, y)
# predictions = model.predict(X)

# # Evaluate the model
# evaluator = my_evaluation(predictions, y)
# print(f"F1 Score: {evaluator.f1()}")
