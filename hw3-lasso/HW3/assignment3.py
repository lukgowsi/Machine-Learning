#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional, Any, Callable, Dict, Union
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score
import random
from typeguard import typechecked

random.seed(42)
np.random.seed(42)


@typechecked
def read_data(filename: str) -> pd.DataFrame:
    """
    Read the data from the filename. Load the data it in a dataframe and return it.
    """
    ########################
    ## Your Solution Here ##
    ########################
    df = pd.read_csv(filename)
    return df


@typechecked
def data_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Follow all the preprocessing steps mentioned in Problem 2 of HW2 (Problem 2: Coding: Preprocessing the Data.)
    Return the final features and final label in same order
    You may use the same code you submiited for problem 2 of HW2
    """
    #######################
    ## Your Solution Here ##
    ########################
    data = df.dropna()
    
    x, y = data.iloc[:, :-1], data.iloc[:, [-1]]
    
    nonNum = x.select_dtypes(exclude = ['int64', 'float64'])
    withNum = x.select_dtypes(include = ['int64', 'float64'])  
    df1 = pd.get_dummies(nonNum['League'], columns = ['League'], prefix = 'League')
    df2 = pd.get_dummies(nonNum['Division'], columns = ['Division'], prefix = 'Division')
    # df3 = pd.get_dummies(nonNum['Player'], columns = ['Player'], prefix = 'Player')
    
    df4 = pd.concat([withNum, df1, df2], axis = 1)  
    
    series = pd.Series(y.iloc[:, 0])
    series1 = series.replace("N", 1)
    series2 = series1.replace("A", 0)
    
    # print(df4)
    # print(series2)
    
    return df4, series2

@typechecked
def data_split(features: pd.DataFrame, label: pd.Series, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split 80% of data as a training set and the remaining 20% of the data as testing set
    return training and testing sets in the following order: X_train, X_test, y_train, y_test
    """
    ########################
    ## Your Solution Here ##
    ########################
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = test_size)
    
    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_test)
    

@typechecked
def train_ridge_regression(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, max_iter: int = int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Ridge Regression, train the model object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {"ridge": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    ########################
    ## Your Solution Here ##
    ########################
    for j in range(len(lambda_vals)):
        # aucs['ridge'].append([])
        garbage = Ridge(alpha = lambda_vals[j], max_iter = max_iter)
        # howdy = []
    
        for i in range(n):
            garbage.fit(x_train, y_train)
            prediction = garbage.predict(x_test)
            actual = roc_auc_score(y_test, prediction)
            # howdy.append(actual)
            aucs["ridge"].append({lambda_vals[j]:actual})
        # print(howdy)

    
    # print(len(aucs["ridge"]))

    print("ridge mean AUCs:")
    ridge_aucs = pd.DataFrame(aucs["ridge"])
    ridge_mean_auc = {}
    for lambda_val, ridge_auc in zip(lambda_vals, ridge_aucs.mean(numeric_only=True)):
        ridge_mean_auc[lambda_val] = ridge_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % ridge_auc)
    return ridge_mean_auc


@typechecked
def train_lasso(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, max_iter=int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Lasso Model, train the object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {"lasso": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    ########################
    ## Your Solution Here ##
    ########################
    for j in range(len(lambda_vals)):
        # aucs['lasso'].append([])
        garbage = Lasso(alpha = lambda_vals[j], max_iter = max_iter)
        # howdy = []
    
        for i in range(n):
            garbage.fit(x_train, y_train)
            prediction = garbage.predict(x_test)
            actual = roc_auc_score(y_test, prediction)
            # howdy.append(actual)
            aucs['lasso'].append({lambda_vals[j]:actual})

    print("lasso mean AUCs:")
    lasso_mean_auc = {}
    lasso_aucs = pd.DataFrame(aucs["lasso"])
    for lambda_val, lasso_auc in zip(lambda_vals, lasso_aucs.mean(numeric_only=True)):
        lasso_mean_auc[lambda_val] = lasso_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % lasso_auc)
    return lasso_mean_auc


@typechecked
def ridge_coefficients(x_train: pd.DataFrame, y_train: pd.Series,optimal_alpha: float, max_iter=int(1e8)) -> Tuple[Ridge, np.ndarray]:
    """
    return the tuple consisting of trained Ridge model with alpha as optimal_alpha and the coefficients
    of the model
    """
    ########################
    ## Your Solution Here ##
    ########################
    bruh = Ridge(alpha = optimal_alpha, max_iter=max_iter)
    bruh.fit(x_train, y_train)
    coefficient = bruh.coef_
    
    return bruh, coefficient


@typechecked
def lasso_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Lasso, np.ndarray]:
    """
    return the tuple consisting of trained Lasso model with alpha as optimal_alpha and the coefficients
    of the model
    """
    ########################
    ## Your Solution Here ##
    ########################
    bruh = Lasso(alpha = optimal_alpha, max_iter=max_iter)
    bruh.fit(x_train, y_train)
    coefficient = bruh.coef_
    
    return bruh, coefficient


@typechecked
def ridge_area_under_curve(model_R, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    return area under the curve measurements of trained Ridge model used to find coefficients,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    ########################
    ## Your Solution Here ##
    ########################
    heh = model_R.predict(x_test)
    area = roc_auc_score(y_test, heh)

    return area

@typechecked
def lasso_area_under_curve(model_L, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    return area under the curve measurements of Lasso Model,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    ########################
    ## Your Solution Here ##
    ########################
    heehee = model_L.predict(x_test)
    area = roc_auc_score(y_test, heehee)

    return area    


class Node:
    @typechecked
    def __init__(self, split_val: float, data: Any = None, left: Any = None, right: Any = None,) -> None:
        if left is not None:
            assert isinstance(left, Node)

        if right is not None:
            assert isinstance(right, Node)

        self.left = left
        self.right = right
        self.split_val = split_val  # value (of a variable) on which to split. For leaf nodes this is label/output value
        self.data = data  # data can be anything! we recommend dictionary with all variables you need


class TreeRegressor:
    @typechecked
    def __init__(self, data: np.ndarray, max_depth: int) -> None:
        self.data = (
            data  # last element of each row in data is the target variable
        )
        self.max_depth = max_depth  # maximum depth
        # YOU MAY ADD ANY OTHER VARIABLES THAT YOU NEED HERE
        # YOU MAY ALSO ADD FUNCTIONS **WITHIN CLASS or functions INSIDE CLASS** TO HELP YOU ORGANIZE YOUR BETTER
        ## YOUR CODE HERE

    @typechecked
    def build_tree(self) -> Node:
        """
        Build the tree
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        # X, Y = self.data[:,:-1], self.data[:,-1]
        # sample, features = np.shape(X)
        
        # # split until stopping conditions are met
        # if sample>=self.split_val and depth<=self.max_depth:
        #     best_split = self.get_best_split(data, sample, features)
        #     if best_split["data"]>0:
        #         left = self.build_tree(best_split["data"], curr_depth+1)
        #         right = self.build_tree(best_split["data"], curr_depth+1)
        #         # return decision node
        #         return Node(best_split["data"], best_split["split_va;"], 
        #                     left, right, best_split["data"])
        
        # leaf = Node(Y)
        # # return leaf node
        # return leaf

        
        
        

    @typechecked
    def mean_squared_error(self, left_split: np.ndarray, right_split: np.ndarray) -> float:
        """
        Calculate the mean squared error for a split dataset
        left split is a list of rows of a df, rightmost element is label
        return the sum of mse of left split and right split
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        # yLeft = np.array([])
        # yRight = np.array([])
        
        # for i in range(len(left_split)):
        #     np.append(yLeft, left_split[i][1])
        # average1 = yLeft.mean()
        # sum1 = ((yLeft - average1)**2) / len(left_split)
        
        
        # for j in range(len(right_split)):
        #     np.append(yRight, right_split[j][1])
        # average2 = yRight.mean()
        # sum2 = ((yRight - average2)**2) / len(right_split)
        
        # answer = sum1 + sum2
        
        # return answer
        

    @typechecked
    def split(self, node: Node, depth: int) -> None:
        """
        Do the split operation recursively
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        # data_left = np.array([row for row in data if row[split_value]<=threshold])
        # data_right = np.array([row for row in data if row[split_value]>threshold])
        # return dataset_left, dataset_right
        
        
        

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset AND create a Node
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        # dict to store the best split
        # best = {}
        # max_info_gain = -float("inf")
        
        # for i in range(len(data)):
        #     feature_values = dataset[:, data[i]]
        #     possible_thresholds = np.unique(feature_values)
        #     # loop over all the feature values present in the data
        #     for j in split_val:
        #         data_left, data_right = self.split(data)
        #         if len(data.left)>0 and len(data.right)>0:
        #             y, leftY, rightY = data[:, -1], data_left[:, -1], data_right[:, -1]
        #             # compute information gain
        #             curr_info = self.information(y, leftY, rightY)
        #             if curr_info>max_info:
        #                 best_split["threshold"] = threshold
        #                 best_split["data_left"] = data_left
        #                 best_split["data_right"] = data_right
        #                 best_split["info_gain"] = curr_info
        #                 max_info = curr_info
                        
        # # return best split
        # return best_split
        
        

    @typechecked
    def one_step_split(
        self, index: int, value: float, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dataset based on an attribute and an attribute value
        index is the variable to be split on (left split < threshold)
        returns the left and right split each as list
        each list has elements as `rows' of the df
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        # if self.data < value:
            

@typechecked
def compare_node_with_threshold(node: Node, row: np.ndarray) -> bool:
    """
    Return True if node's value > row's value (of the variable)
    Else False
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    # if node.data > row:
    #     return True
    # else:
    #     return False


@typechecked
def predict(
    node: Node, row: np.ndarray, comparator: Callable[[Node, np.ndarray], bool]
) -> float:
    ######################
    ### YOUR CODE HERE ###
    ######################
    # preditions = [self.make_prediction(x, self.root) for x in X]
    # return preditions
    pass


class TreeClassifier(TreeRegressor):
    def build_tree(self):
        ## Note: You can remove this if you want to use build tree from Tree Regressor
        ######################
        ### YOUR CODE HERE ###
        ######################
        pass

    @typechecked
    def gini_index(self, left_split: np.ndarray, right_split: np.ndarray, classes: List[float]) -> float:
        """
        Calculate the Gini index for a split dataset
        Similar to MSE but Gini index instead
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        # class_labels = np.unique(y)
        # gini = 0
        # for cls in class_labels:
        #     p_cls = len(y[y == cls]) / len(y)
        #     gini += p_cls**2
        # return 1 - gini

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset
        """
        # classes = list(set(row[-1] for row in data))
        ######################
        ### YOUR CODE HERE ###
        ######################
        


if __name__ == "__main__":
    # Question 1
    filename = ""  # Provide the path of the dataset
    df = read_data('hitters.csv')
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    max_iter = 1e8
    final_features, final_label = data_preprocess(df)
    x_train, x_test, y_train, y_test = data_split(
        final_features, final_label, 0.2
    )
    ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)
    ridge_auc = ridge_area_under_curve(model_R, x_test, y_test)

    # Plot the ROC curve of the Ridge Model. Include axes labels,
    # legend and title in the Plot. Any of the missing
    # items in plot will result in loss of points.
    ########################
    ## Your Solution Here ##
    ########################
    y_pred = model_R.predict(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label='Ridge')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('ridge auc')
    plt.clf()
    

    lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)

    # Plot the ROC curve of the Lasso Model.
    # Include axes labels, legend and title in the Plot.
    # Any of the missing items in plot will result in loss of points.
    ########################
    ## Your Solution Here ##
    ########################
    y_pred = model_L.predict(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label='Lasso')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('lasso auc')
    plt.clf()

    # SUB Q1
    data_regress = np.loadtxt('noisy_sin_subsample_2.csv', delimiter=",")
    data_regress = np.array([[x, y] for x, y in zip(*data_regress)])
    plt.figure()
    plt.scatter(data_regress[:, 0], data_regress[:, 1])
    plt.xlabel("Features, x")
    plt.ylabel("Target values, y")
    plt.show()

    mse_depths = []
    for depth in range(1, 5):
        regressor = TreeRegressor(data_regress, depth)
        tree = regressor.build_tree()
        mse = 0.0
        for data_point in data_regress:
            mse += (
                data_point[1]
                - predict(tree, data_point, compare_node_with_threshold)
            ) ** 2
        mse_depths.append(mse / len(data_regress))
    plt.figure()
    plt.plot(mse_depths)
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.show()

    # SUB Q2
    csvname = "new_circle_data.csv"  # Place the CSV file in the same directory as this notebook
    data_class = np.loadtxt(csvname, delimiter=",")
    data_class = np.array([[x1, x2, y] for x1, x2, y in zip(*data_class)])
    plt.figure()
    plt.scatter(
        data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap="bwr"
    )
    plt.xlabel("Features, x1")
    plt.ylabel("Features, x2")
    plt.show()

    accuracy_depths = []
    for depth in range(1, 8):
        classifier = TreeClassifier(data_class, depth)
        tree = classifier.build_tree()
        correct = 0.0
        for data_point in data_class:
            correct += float(
                data_point[2]
                == predict(tree, data_point, compare_node_with_threshold)
            )
        accuracy_depths.append(correct / len(data_class))
    # Plot the MSE
    plt.figure()
    plt.plot(accuracy_depths)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()
