import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, log_loss
import matplotlib.pyplot as plt
import random
from typeguard import typechecked


random.seed(42)
np.random.seed(42)

################
################
## Q1
################
################

@typechecked
def read_classification_data(file_path:str) -> (np.array, np.array):
  '''
    Read the data from the path.
    Return the data as 2 np arrays each with shape (number_of_rows_in_dataframe, 1)
    Order (np.array from first row), (np.array from second row)
  '''
  ########################
  ## Your Solution Here ##
  ########################
  
  df = pd.read_csv(file_path, header=None)
    
  # extract the first two rows as numpy arrays
  row1 = np.array(df.iloc[0]).reshape(-1, 1)
  row2 = np.array(df.iloc[1]).reshape(-1, 1)
    
  print(row1)
  print(row2)  
  
  return row1, row2
  

@typechecked
def sigmoid(s:np.array) -> np.array:
  '''
    Return the sigmoid of every number in the array s as an array of floating point numbers
    sigmoid(s)= 1/(1+e^(-s))
  '''
  ########################
  ## Your Solution Here ##
  ########################
  
  sigmoids = 1/(1 + np.exp(-s))
  
  return sigmoids

@typechecked
def cost_function(w:float, b:float, X:np.array, y:np.array) -> float:
  '''
  Inputs definitions:
    w : weight
    b : bias
    X : input  with shape (number_of_rows_in_dataframe, 1)
    y : target with shape (number_of_rows_in_dataframe, 1)
  Return the loss as a float data type. 
  '''
  ########################
  ## Your Solution Here ##
  ########################
  
  num = X.shape[0]
  
  line = np.dot(X, w) + b
  
  y1 = sigmoid(line)
  
  loss = -(1/num) * np.sum(y * np.log(y1) + (1 - y) * np.log(1 - y1))
  
  return loss

@typechecked
def cross_entropy_optimizer(w:float, b:float, X:np.array, y:np.array, num_iterations:int, alpha:float) -> (float, float, list):
  '''
    Inputs definitions:
      w              : initial weight
      b              : initial bias
      X              : input  with shape (number_of_rows_in_dataframe, 1)
      y              : target with shape (number_of_rows_in_dataframe, 1)
      num_iterations : number of iterations 
      alpha          : Learning rate

    Task: Iterate for given number of iterations and find optimal weight and bias 
    while also noting the change in cost/ loss after every iteration

    Make use of the cost_function() above

    Return (updated weight, updated bias, list of "costs" after each iteration) in this order
    "costs" list contains float type numbers  
  '''
  ########################
  ## Your Solution Here ##
  ########################
  
  n = X.shape[0]
  costs = []
  
  for i in range(num_iterations):
    line = np.dot(X, w) + b
    y1 = sigmoid(line)
    
    #calculate loss
    cost = cost_function(w,b,X,y)
    costs.append(cost)
    
    diff = y1 - y
    newWeight = np.dot(X.T, diff) / n
    newBias = np.sum(diff) / n
    
    #update
    w = float(w - alpha*newWeight)
    b = float(b - alpha*newBias)
    
  return w, b, costs
    
    
    

################
################
## Q3 a
################
################

@typechecked
def read_sat_image_data(file_path:str) -> pd.DataFrame:
  '''
    Input: filepath to a .csv file
    Output: Return a DataFrame with the data from the given csv file 
  '''
  ########################
  ## Your Solution Here ##
  ########################

@typechecked
def remove_nan(df : pd.DataFrame) -> pd.DataFrame:
  '''
    Remove nan values from the dataframe and return it
  '''
  ########################
  ## Your Solution Here ##
  ########################

@typechecked
def normalize_data(Xtrain : pd.DataFrame, Xtest : pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
  '''
    Normalize each column of the dataframes and Return the dataframes
    Use sklearn.preprocessing.StandardScaler library to normalize
    Return the results in the order Xtrain_norm, Xtest_norm
  '''
  ########################
  ## Your Solution Here ##
  ########################

@typechecked
def labels_to_binary(y : pd.DataFrame) -> pd.DataFrame:
  '''
  Make the lables [1,2,3,4,5] as 0 and [6] as 1
  Return the DataFrame 
  '''
  ########################
  ## Your Solution Here ##
  ########################


################
################
## Q3 b
################
################

@typechecked
def cross_validate_c_vals(X : pd.DataFrame, y : pd.DataFrame, n_folds: int, c_vals : np.array, d_vals: np.array) -> (np.array, np.array):
  '''
    Return the matrices (ERRAVGdc, ERRSTDdc) in the same order
    More details about the imlementation are provided in the main function
  '''
  ########################
  ## Your Solution Here ##
  ########################

@typechecked
def plot_cross_val_err_vs_c(ERRAVGdc: np.array, ERRSTDdc: np.array, c_vals: np.array, d_vals: np.array) ->None:
  '''
   Please write the code in below block to generate the graphs as described in the question.
   Note that the code will not be graded, but the graphs submitted in the report will be evaluated.
  '''
  ########################
  ## Your Solution Here ##
  ########################

################
################
## Q3 c 
################
################
@typechecked
def evaluate_c_d_pairs(X_train : pd.DataFrame, y_train : pd.DataFrame, X_test : pd.DataFrame, y_test : pd.DataFrame, n_folds : int, c_vals : np.array, d_vals: np.array) ->(np.array, np.array, np.array, np.array):
  '''
    Return in the order: ERRAVGdcTEST, SuppVect, vmd, MarginT
    More details about the imlementation are provided in the main function
    Shape:
      ERRAVGdcTEST = np array with shape len(d_vals)
      SuppVect     = np array with shape len(d_vals)
      vmd          = np array with shape len(d_vals)
      MarginT      = np array with shape len(d_vals)
  '''
  ########################
  ## Your Solution Here ##
  ########################

@typechecked
def plot_test_errors(ERRAVGdcTEST : np.array, d_vals : np.array) -> None:
  '''
   Please write the code in below block to generate the graphs as described in the question.
   Note that the code will not be graded, but the graphs submitted in the report will be evaluated.
  '''
  ########################
  ## Your Solution Here ##
  ########################

################
################
## Q3 d
################
################
@typechecked
def plot_avg_support_vec(SuppVect : np.array, d_vals : np.array) ->None:
  '''
   Please write the code in below block to generate the graphs as described in the question.
   Note that the code will not be graded, but the graphs submitted in the report will be evaluated.
  '''
  ########################
  ## Your Solution Here ##
  ########################

@typechecked
def plot_avg_violating_support_vec(vmd : np.array, d_vals : np.array) ->None:
  '''
   Please write the code in below block to generate the graphs as described in the question.
   Note that the code will not be graded, but the graphs submitted in the report will be evaluated.
  '''
  ########################
  ## Your Solution Here ##
  ########################

################
################
## Q3 e
################
################
@typechecked
def plot_avg_hyperplane_margins(MarginT : np.array, d_vals : np.array) ->None:
  '''
   Please write the code in below block to generate the graphs as described in the question.
   Note that the code will not be graded, but the graphs submitted in the report will be evaluated.
  '''
  ########################
  ## Your Solution Here ##
  ########################

if __name__ == "__main__":
  '''
  General Instructions:
    If you want to use a library which is not already included at the top of this file,
    import them in the function in which you are using the library, not at the top of this file.
    If you import it at the top of this file, your code will not be evaluated correctly by the autograder.
    You will not be awarded any points if your code fails because of this reason.
  '''

  ################
  ################
  ## Q1
  ################
  ################
  '''
    Below we load the data.
    Provide the path for data.
    Implement- read_classification_data()
  '''
  ########################
  ## Your Solution Here ##
  classification_data_2d_path="2d_classification_data_entropy.csv"
  ########################
  x,y= read_classification_data(classification_data_2d_path)

  '''
    Below code initializes the weight and bias to 1, then iterates 300 times to find a better fit. 
    The cost/error is plotted against the number of iterations. 
    Please submit a screenshot of the plot in the report to receive points. 
    Implement- sigmoid(), cost_function(), cross_entropy_optimizer()
  '''
  w = 1
  b = 1
  num_iterations = 300
  w,b, costs = cross_entropy_optimizer(w,b,x,y,num_iterations,0.1)
  print("Weignt W: ",w)
  print("Bias b: ", b)
  # costs = costs.reshape((300,))
  # print(type(num_iterations))
  # print(type(costs))
  # print(num_iterations)
  # print(len(costs))
  plt.plot(range(num_iterations),costs)
  plt.savefig('plot.png')

  ################
  ################
  ## Q3 a
  ################
  ################

  ########################
  '''
    Below we load the data into dataframe.
    Provide the path for training and test data.
    Implement- read_sat_image_data()
  '''
  ########################
  ## Your Solution Here ##
  sat_image_Training_path = ""
  sat_image_Test_path     = ""
  ########################

  train_df = read_sat_image_data(sat_image_Training_path)#Training set
  test_df  = read_sat_image_data(sat_image_Test_path)#Testing set

  '''
    Below code 
      -removes nan values from data frame
      -loads the train and test dataframes
      -Normalize the input dataframes
      -convert labels to binary
    Implement- remove_nan(), normalize_data(), labels_to_binary()
  '''
  train_df_nan_removed= remove_nan(train_df)
  test_df_nan_removed= remove_nan(test_df)

  ytrain = train_df_nan_removed[['Class']]
  Xtrain = train_df_nan_removed.drop(['Class'], axis=1)
  
  ytest = test_df_nan_removed[['Class']]
  Xtest = test_df_nan_removed.drop(['Class'], axis=1)
  
  Xtrain_norm, Xtest_norm = normalize_data(Xtrain, Xtest)

  ytrain_bin_label = labels_to_binary(ytrain)
  ytest_bin_label  = labels_to_binary(ytest)

  ################
  ################
  ## Q3 b
  ################
  ################
  '''
    ERRAVGdc is a matrix with ERRAVGdc[c][d] = "Average Mean Absolute Error" of 10 folds for 'C'=c and degree='d'
    ERRSTDdc is a matrix with ERRSTDdc[c][d] = "Standard Deviation" of 10 folds for 'C'=c and degree='d'
    Both the matrices have size (len(c_vals), len(d_vals))
    Fill these matrices in the cross_validate_c_vals function
    For each 'c' and 'd' values :
      Split the data into 10 folds and for each fold:
          Find the predictions and corresponding mean Absolute errors and store the error
      Evaluate the "Average Mean Absolute Error" and "Standard Deviation" from stored errors
      Update the ERRAVGdc[c][d], ERRSTDdc[c][d] with the evaluated "Average Mean Absolute Error" and "Standard Deviation"
        
    Note: 'C' is the trade-off constant, which controls the trade-off between a smooth decision boundary and classifying the training points correctly.
    Note: 'degree' is the degree of the polynomial kernel used with the SVM
     Matrices ERRAVGdc, ERRSTDdc look like this:
              d=1   d=2   d=3   d=4
    --------- ---   ---   ---   --- 
    c=0.01 | .     .     .     .
    c=0.1  | .     .     .     .
    c=1    | .     .     .     .
    c=10   | .     .     .     .
    c=100  | .     .     .     .

    Implement- cross_validate_c_vals(), plot_cross_val_err_vs_c()
  '''
  c_vals  = np.power(float(10), range(-2, 2 + 1))
  n_folds = 5
  d_vals  = np.array([1,2,3,4])

  ERRAVGdc, ERRSTDdc = cross_validate_c_vals(Xtrain_norm, ytrain_bin_label, n_folds, c_vals, d_vals)

  plot_cross_val_err_vs_c(ERRAVGdc, ERRSTDdc, c_vals, d_vals)

  ################
  ################
  ## Q3 c
  ################
  ################
  d_vals=[1,2,3,4]
  n_folds = 5
  '''
    Use the results from above and Fill the best c values for d=1,2,3,4
  '''
  ########################
  ## Your Solution Here ##
  new_c_vals = []
  ########################

  '''
  Below are the vectors evaluated by evaluate_c_d_pairs() function
    ERRAVGdcTEST - Average Testing error for each value of 'd'
    SuppVect     - Average Number of Support Vectors for each value of 'd'
    vmd          - Average Number of Support Vectors that Violate the Margin for each value of 'd'
    MarginT      - Average Value of Hyperplane Margins for each value of 'd'
  Implement- evaluate_c_d_pairs(), plot_test_errors, plot_avg_support_vec(), plot_avg_violating_support_vec(), plot_avg_hyperplane_margins()
  '''
  
  ERRAVGdcTEST, SuppVect, vmd, MarginT = evaluate_c_d_pairs(Xtrain_norm, ytrain_bin_label, Xtest_norm, ytest_bin_label, n_folds, new_c_vals, d_vals)
  plot_test_errors(ERRAVGdcTEST, d_vals)
  
  ################
  ################
  ## Q3 d
  ################
  ################
  plot_avg_support_vec(SuppVect, d_vals)
  plot_avg_violating_support_vec(vmd, d_vals)
  
  ################
  ################
  ## Q3 e
  ################
  ################
  plot_avg_hyperplane_margins(MarginT, d_vals)