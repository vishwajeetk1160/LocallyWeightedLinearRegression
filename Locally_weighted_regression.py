import numpy as np
import pandas as pd
import math


dataset=pd.read_csv('Dataset1.csv')
row_index=input("please enter the index number you want to choose as test point")
row_index=int(row_index)
target=dataset.loc[row_index][1:6]

def gaussian_kernel(x, x0, c, a=1.0):
    x=np.mat(x)
    diff = x - x0
    dot_product = diff * diff.T
    dot_product_sum=dot_product.sum()
    dot_product_sqrt=math.sqrt(dot_product_sum)
    return a * np.exp(dot_product_sqrt / (-2.0 * c**2))


def get_weights(training_inputs, datapoint, c=1.0):
    x = np.mat(training_inputs)
    n_rows = x.shape[0]
    weights = np.mat(np.eye(n_rows))
    for i in range(n_rows):
        weights[i, i] = gaussian_kernel(datapoint, x[i], c)

    return weights


def lwr_predict(training_inputs, training_outputs, datapoint, c=1.0):
    weights = get_weights(training_inputs, datapoint, c=c)
    x = np.mat(training_inputs)
    y = np.mat(training_outputs).T
    xt = x.T * (weights * x)
    betas = xt.I * (x.T * (weights * y.T))
    datapoint=np.mat(datapoint)
    return datapoint * betas

def get_25_nearest_points(target_1, dataset_1):
	difference_list=[]
	for i in range(len(dataset_1)):
		diff=target_1 - dataset_1.loc[i][1:6]
		dot_product = (diff * diff.T)
		dot_product_sum=dot_product.sum()
		dot_product_sqrt=math.sqrt(dot_product_sum)
		difference_list.append(dot_product_sqrt)
	
	sorted_index=np.argsort(difference_list)
	sorted_index_1=sorted_index[1:26]
	return dataset_1.iloc[sorted_index_1,:]


var_2= get_25_nearest_points(target, dataset)

x_value=var_2.iloc[:,1:6]
y_value=var_2.iloc[:, 6:7]

var_answer=lwr_predict(x_value, y_value, target)
print ("predicted value")
print (var_answer)
print ("actual value")
print (float (dataset.loc[row_index][6:7]))





