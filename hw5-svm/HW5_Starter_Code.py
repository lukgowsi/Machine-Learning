#!/usr/bin/env python
# coding: utf-8

print("Acknowledgment:")
print("https://github.com/pritishuplavikar/Face-Recognition-on-Yale-Face-Dataset")

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import glob
from numpy import linalg as la
import random
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import sklearn
from typing import Tuple, List
from typeguard import typechecked


@typechecked
def qa1_load(folder_path:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the dataset (tuple of x, y the label).

    x should be of shape [165, 243 * 320]
    label can be extracted from the subject number in filename. ('subject01' -> '01 as label)
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    x = np.zeros((165, 243 * 320))
    y = np.zeros((165,))

    filenames = os.listdir(folder_path)
    filenames.sort()
    print(filenames)

    for i, filename in enumerate(filenames):
        img = mpimg.imread(os.path.join(folder_path, filename))
        x[i] = img.flatten()
        y[i] = int(filename.split(".")[0][-2:])

    return x, y

@typechecked
def qa2_preprocess(dataset:np.ndarray) -> np.ndarray:
    """
    returns data (x) after pre processing

    hint: consider using preprocessing.MinMaxScaler
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    scaler = MinMaxScaler()
    x = scaler.fit_transform(dataset)
    return x

@typechecked
def qa3_calc_eig_val_vec(dataset:np.ndarray, k:int)-> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Calculate eig values and eig vectors.
    Use PCA as imported in the code to create an instance
    return them as tuple PCA, eigen_value, eigen_vector
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    pca = PCA(n_components=k)
    pca.fit(dataset)
    eVals = pca.explained_variance_
    eVecs = pca.components_
    return pca, eVals, eVecs

def qb_plot_written(eig_values:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    # cumulated sum of values
    cSum = np.cumsum(eig_values)
    print(cSum)
    # total sum of values
    totSum = np.sum(eig_values)
    print(totSum)
    # find energy of the values
    energy = cSum / totSum
    # plot the curve of eigenvalues vs. energy
    plt.plot(np.arange(len(eig_values))+1, energy)
    plt.xlabel('Number of components k')
    plt.ylabel('Energy')
    plt.title('Energy vs. Numbers k')
    plt.savefig('fig1.png')
    plt.close()

    # find number of components with 50% energy
    n = np.argmax(energy >= 0.5) + 1
    print("Number of components needed to capture 50 percent of the energy:", n)
    

@typechecked
def qc1_reshape_images(pca:PCA, dim_x = 243, dim_y = 320) -> np.ndarray:
    """
    reshape the pca components into the shape of original image so that it can be visualized
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    components = pca.components_
    eigenFaces = components.reshape(-1, dim_x, dim_y)

    return eigenFaces

def qc2_plot(org_dim_eig_faces:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        ax[row][col].imshow(org_dim_eig_faces[i], cmap='gray')
        ax[row][col].set_title(f"Eigenface {i+1}")
        ax[row][col].axis('off')
    plt.tight_layout()
    plt.savefig('fig2.png')
    plt.close()

@typechecked
def qd1_project(dataset:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the projection of the dataset 
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    return pca.transform(dataset)

@typechecked
def qd2_reconstruct(projected_input:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the reconstructed image given the pca components
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    return pca.inverse_transform(dataset)

def qd3_visualize(dataset:np.ndarray, pca:PCA, dim_x = 243, dim_y = 320):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots. You can use other functions that you coded up for the assignment
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qe1_svm(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold).

    Hint: you can pick 5 `k' values uniformly distributed
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qe2_lasso(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold) in that order.

    Hint: you can pick 5 `k' values uniformly distributed
    """
    ######################
    ### YOUR CODE HERE ###
    ######################



if __name__ == "__main__":

    faces, y_target = qa1_load("./data/")
    dataset = qa2_preprocess(faces)
    pca, eig_values, eig_vectors = qa3_calc_eig_val_vec(dataset, len(dataset))

    qb_plot_written(eig_values)

    num = len(dataset)
    org_dim_eig_faces = qc1_reshape_images(pca)
    qc2_plot(org_dim_eig_faces)

    qd3_visualize(dataset, pca)
    best_k, result = qe1_svm(dataset, y_target, pca)
    print(best_k, result)
    best_k, result = qe2_lasso(dataset, y_target, pca)
    print(best_k, result)
