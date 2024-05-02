import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion_matrix = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        correct = self.predictions == self.actuals
        self.acc = float(Counter(correct)[True]) / len(correct)
        self.confusion_matrix = {}
        for label in self.classes_:
            tp = np.sum((self.actuals == label) & (self.predictions == label))
            tn = np.sum((self.actuals != label) & (self.predictions != label))
            fp = np.sum((self.actuals != label) & (self.predictions == label))
            fn = np.sum((self.actuals == label) & (self.predictions != label))
            self.confusion_matrix[label] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        return

    def accuracy(self):
        if self.confusion_matrix == None:
            self.confusion()
        return self.acc

    def precision(self, target=None, average="macro"):
        if self.confusion_matrix == None:
            self.confusion()
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp + fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            if average == "micro":
                prec = self.accuracy()
            else:
                prec = 0
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    if average == "macro":
                        prec += prec_label
                    elif average == "weighted":
                        prec += prec_label * (self.confusion_matrix[label]["TP"] + self.confusion_matrix[label]["FP"]) / len(self.actuals)
                if average == "macro":
                    prec /= len(self.classes_)
        return prec

    def recall(self, target=None, average="macro"):
        if self.confusion_matrix == None:
            self.confusion()

        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if tp + fn == 0:
                rec = 0
            else:
                rec = float(tp) / (tp + fn)
        else:
            if average == "micro":
                rec = self.accuracy()
            else:
                rec = 0
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        rec_label = 0
                    else:
                        rec_label = float(tp) / (tp + fn)
                    if average == "macro":
                        rec += rec_label
                    elif average == "weighted":
                        rec += rec_label * (self.confusion_matrix[label]["TP"] + self.confusion_matrix[label]["FN"]) / len(self.actuals)
                if average == "macro":
                    rec /= len(self.classes_)
        return rec

    def f1(self, target=None, average="macro"):
        if self.confusion_matrix == None:
            self.confusion()

        prec = self.precision(target, average)
        rec = self.recall(target, average)

        if prec + rec == 0:
            f1_score = 0
        else:
            f1_score = 2 * (prec * rec) / (prec + rec)

        return f1_score

