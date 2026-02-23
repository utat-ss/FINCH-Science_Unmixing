import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, RationalQuadratic
import math

#getting the data to train on
file = pd.read_csv('simpler_data.csv')
#getting y's   
y = []
for i in range(len(file["Spectra"])):
    y.append([file.loc[i,"gv_fraction"],file.loc[i,"npv_fraction"],file.loc[i,"soil_fraction"]])

#getting x's
x = []
for i in range(len(file["Spectra"])):
    newAr = []
    for j in range(900,1710,10):
        newAr.append(file.loc[i,str(j)])
    x.append(newAr)
print(len(x))


#spliting the data
def k_split_dataset(X: int, Y: int, train_size_percentage: int) -> list[list[list]]:
    """Return kfold splits
    
    """
    kfold = KFold(n_splits=train_size_percentage, shuffle= True, random_state=42)

    ids = range(len(X))
    train_ids = np.array([])
    test_ids = np.array([])
    data = []
    for train, test  in kfold.split(ids):
        train_ids = np.array([])
        test_ids = np.array([])
        x_train = []
        x_test = [] 
        y_train = []  
        y_test = []
        train_ids = np.append(train,train_ids)
        test_ids = np.append(test,test_ids)
        for i in train_ids.astype(int):

            x_train.append(x[i])
            y_train.append(y[i])
        for i in test_ids.astype(int):
            x_test.append(x[i])
            y_test.append(y[i])
        data.append([x_train, x_test, y_train, y_test])
    return data
#orig_scores = cross_validation_scores(x,y,len(x)//10)


def get_all_splits(first: int, last: int, ar: list) -> None:
    for k in range(first, last):
        ar.append(k_split_dataset(x,y,k))

r2s = []
all_splits = []
get_all_splits(2, 20, all_splits)

def find_the_most_optimal_k_splits(all_splits: list[int], r2s: list[int]) -> None:
   """Find r^2 for different splitting using kfold. However, choose the best one among splits with n folds.
  As a regression function, there can be any other
  """
    splitnum = 2
    for i in all_splits:
        bestr2 = -1
        bestrmse = -1
        best_y_test = []
        best_y_predict = []
        for j in i:
                #splitting the dataset
            X_train, X_test, y_train, y_test = j[0], j[1], j[2], j[3]
                #choosing regression type and training
            regr = GaussianProcessRegressor(kernel=RationalQuadratic(), random_state=42)
            regr.fit(X_train, y_train)
            y_predict = regr.predict(X_test)
                #getting results
            r2 = r2_score(y_test,y_predict)
            if r2> bestr2:
                bestr2 = r2
                best_y_test = y_test
                best_y_predict = y_predict
            print("train_size: "+str(splitnum)+", r2: "+str(r2))
        print(len(X_train), len(X_test))
        r2s.append(bestr2)
        splitnum+=1

def find_randomsplit(r2s: list[int]) -> None:
  """Find r^2 for different splitting using randomsplit
  As a regression function, there can be any other
  """
  
    for i in range(10,100,5):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1-i/100, train_size=i/100, random_state=42)
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        y_predict = regr.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_predict)
        r2 = r2_score(y_test,y_predict)
        r2s.append(r2)
        print("train_size: "+str(i)+"%, r2: "+str(r2)+", rmse: "+str(rmse))


find_the_most_optimal_k_splits(all_splits, r2s)
plt.rcParams["font.family"] = "Times New Roman"

csfont = {'fontname':'Times New Roman'}
plt.plot(range(2,20),r2s)
plt.title("R^2 vs Train Size", **csfont)
plt.xlabel("Train Size(# of splits)", **csfont)
plt.ylabel("R^2", **csfont)
plt.grid(True)
plt.tight_layout()
plt.show()
