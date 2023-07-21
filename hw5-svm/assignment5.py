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
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV
from sklearn.model_selection import GridSearchCV, KFold
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
    # print(filenames)

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
    # cSum = np.cumsum(eig_values)
    # print(cSum)
    # # total sum of values
    # totSum = np.sum(eig_values)
    # print(totSum)
    # # find energy of the values
    # energy = cSum / totSum
    # plot the curve of eigenvalues vs. energy
    # plt.plot(np.arange(len(eig_values))+1, energy)
    # plt.xlabel('Number of components k')
    # plt.ylabel('Energy')
    # plt.title('Energy vs. Numbers k')
    # plt.savefig('fig1.png')
    # plt.close()

    # find number of components with 50% energy
    # n = np.argmax(energy >= 0.5) + 1
    # print("Components k needed for 50 percent of energy:", n)
    

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
    # fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    # for i in range(10):
    #     row, col = i // 5, i % 5
    #     ax[row][col].imshow(org_dim_eig_faces[i], cmap='gray')
    #     ax[row][col].set_title(f"Eigenface {i+1}")
    #     ax[row][col].axis('off')
    # plt.tight_layout()
    # plt.savefig('fig2.png')
    # plt.close()

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
    return pca.inverse_transform(projected_input)


def qd3_visualize(dataset:np.ndarray, pca:PCA, dim_x = 243, dim_y = 320):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots. You can use other functions that you coded up for the assignment
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    print(pca.components_)
    img_1 = dataset[0]
    img_2 = dataset[3]
    img_3 = dataset[6]
    img_4 = dataset[9]
    imgs = np.array([img_1, img_2, img_3, img_4] )
    components = np.array([1, 10, 20, 30, 40, 50] )

    reconstructed_images = []
    for k in components: 
        pca.components_ = pca.components_[:k]
        projected_images = qd1_project(imgs, pca)
        reconstructed_images.append(qd2_reconstruct(projected_images, pca))

    #plot
    num_plots = len(components)
    fig, axs = plt.subplots(num_plots, 4, figsize=(10, 10))
    
    for i, k in enumerate(components):
        
        axs[i, 0].imshow(reconstructed_images[i][0].reshape(243,320), cmap='gray')
        axs[i, 0].set_title(f'Image 1, k={k}')
        axs[i, 1].imshow(reconstructed_images[i][1].reshape(243,320), cmap='gray')
        axs[i, 1].set_title(f'Image 2, k={k}')
        axs[i, 2].imshow(reconstructed_images[i][2].reshape(243,320), cmap='gray')
        axs[i, 2].set_title(f'Image 3, k={k}')
        axs[i, 3].imshow(reconstructed_images[i][2].reshape(243,320), cmap='gray')
        axs[i, 3].set_title(f'Image 4, k={k}')
    
    for ax in axs.flat:
        ax.axis('off')
    
    fig.tight_layout()
        
    
    plt.savefig('fig3.png')
    

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
    skf = StratifiedKFold(n_splits=5)

    kVals = np.linspace(10, 100, 5, dtype=int)

    testAcc = []
    for k in kVals:
        projected = pca.transform(trainX)[:, :k]

        foldAcc = []
        for trainInd, testInd in skf.split(projected, trainY):
            X_train, X_test = projected[trainInd], projected[testInd]
            y_train, y_test = trainY[trainInd], trainY[testInd]

            svm = SVC(kernel='rbf')
            svm.fit(X_train, y_train)

            accuracy = svm.score(X_test, y_test)
            foldAcc.append(accuracy)

        testAcc.append(np.mean(foldAcc))

    #best k value is highest accuracy
    bestK = kVals[np.argmax(testAcc)]
    print(bestK)
    
    #stores highest accuracy with best k value
    maxAcc = max(testAcc)
    print(maxAcc)

    return int(bestK), float(maxAcc)

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
    # skf = StratifiedKFold(n_splits=5)
    # kVals = np.linspace(10, 100, 5, dtype=int)
    
    kVals = np.linspace(10, 100, 5, dtype=int)
    testAcc = []

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

    # project data using PCA
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    for k in kVals:
        # train Lasso with 5 fold cross validation
        lasso = LassoCV(cv=KFold(n_splits=5, shuffle=True, random_state=42), max_iter=10000).fit(X_train_pca[:, :k], y_train)
        accuracy = lasso.score(X_test_pca[:, :k], y_test)

        testAcc.append(accuracy)

    print(kVals)
    print(testAcc)
    # find index of best k value with on test accuracy
    bestKInd = np.argmax(testAcc)
    bestKVal = kVals[bestKInd]
    kAccuracy = testAcc[bestKInd]
    print(bestKVal)
    print(kAccuracy)

    return int(bestKVal), float(kAccuracy)


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