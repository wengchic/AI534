# Logistic regression with L2 and L1 regularizations
# By CHI-CHIEH WENG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

# Loads a data file from a provided file location.
def load_data(path):
    loaded_data = pd.read_csv(path)
    return loaded_data

# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.

def preprocess_data(data):
    normalize = ['Annual_Premium', 'Age', 'Vintage']
    mean = [0]*data.shape[1]
    std = [0]*data.shape[1]

    for i, col in enumerate(data.columns):
        if(col == 'Annual_Premium' or col == 'Age' or col == 'Vintage'):
            
            mean[i] = data[col].mean()
            std[i] = data[col].std()
            data[col] = (data[col] - mean[i]) / std[i]


    return data

# Trains a logistic regression model with L2 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L2_train(train_X, train_Y, val_X, val_Y, lambd):
    # Your code here:
    
    i = 1
    itr = 4000
    learning_rate = 0.01
    top_train_acc = 0
    top_val_acc = 0
    min_loss = float('inf')

    weights = np.zeros(train_X.shape[1])
    
    
    while i < itr:
        i += 1

        # normal gradient without L2 norm
        normal_gradient = ((train_X.multiply(train_Y - sigmoid(train_X.dot(weights)), axis=0)).mean())
        weights = weights + learning_rate * normal_gradient

        # L2 norem contribution excluding w0(dummy feature)
        weights[1:] = weights[1:] - learning_rate * lambd * weights[1:]

        train_acc = threshold(sigmoid(train_X.dot(weights)), train_Y)
        #print(train_acc)
        

        val_acc = threshold(sigmoid(val_X.dot(weights)), val_Y)
        
        top_train_acc = max(top_train_acc, train_acc)
        top_val_acc = max(top_val_acc, val_acc)

        loss = (((-1 * train_Y) * np.log(sigmoid(train_X.dot(weights)))) - ((np.ones(train_X.shape[0]) - train_Y) * np.log(np.ones(train_X.shape[0]) - sigmoid(train_X.dot(weights))))).mean() + lambd * np.sum(np.power(weights[1:], 2))
        min_loss = min(min_loss, loss)

        if np.linalg.norm(normal_gradient) <= 1e-8 or loss == np.nan or loss == float("inf"):
            
            print('break!!!')
            break

    train_acc = threshold(sigmoid(train_X.dot(weights)), train_Y)
    val_acc = threshold(sigmoid(val_X.dot(weights)), val_Y)


    print("itr={}, lr={}, lambda={}, top_train_acc={}, train_acc={}, top_val_acc={}, val_acc={}, min_loss={}".format(i, learning_rate, lambd, top_train_acc, train_acc, top_val_acc, val_acc, min_loss))

    return weights, train_acc, val_acc

# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_X, train_Y, val_X, val_Y, lambd):
    i = 1
    itr = 4000
    learning_rate = 0.01
    top_train_acc = 0
    top_val_acc = 0
    min_loss = float('inf')

    weights = np.zeros(train_X.shape[1])
    
    
    while i < itr:
        i += 1

        # normal gradient without L2 norm
        normal_gradient = ((train_X.multiply(train_Y - sigmoid(train_X.dot(weights)), axis=0)).mean())
        weights = weights + learning_rate * normal_gradient

        # L2 norem contribution excluding w0(dummy feature)
        weights[1:] = np.sign(weights[1:]) * np.maximum(np.abs(weights[1:]) - (learning_rate * lambd), np.zeros(weights[1:].shape))

        train_acc = threshold(sigmoid(train_X.dot(weights)), train_Y)
        #print(train_acc)
        

        val_acc = threshold(sigmoid(val_X.dot(weights)), val_Y)
        

        top_train_acc = max(top_train_acc, train_acc)
        top_val_acc = max(top_val_acc, val_acc)

        loss = (((-1 * train_Y) * np.log(sigmoid(train_X.dot(weights)))) - ((np.ones(train_X.shape[0]) - train_Y) * np.log(np.ones(train_X.shape[0]) - sigmoid(train_X.dot(weights))))).mean() + lambd * np.sum(np.abs(weights[1:]))
        min_loss = min(min_loss, loss)

        if np.linalg.norm(normal_gradient) <= 1e-8 or loss == np.nan or loss == float("inf"):
            
            print('break!!!')
            break

    train_acc = threshold(sigmoid(train_X.dot(weights)), train_Y)
    val_acc = threshold(sigmoid(val_X.dot(weights)), val_Y)


    print("itr={}, lr={}, lambda={}, top_train_acc={}, train_acc={}, top_val_acc={}, val_acc={}, min_loss={}".format(i, learning_rate, lambd, top_train_acc, train_acc, top_val_acc, val_acc, min_loss))

    return weights, train_acc, val_acc

# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda values and then put multiple loss curves in a single plot.
def plot_losses(train_acc_arr, val_acc_arr, lambda_arr):

    #train figure
    x1 = lambda_arr
    y1 = train_acc_arr
    plt.suptitle('Train Accuracy With Different Lambda')
    plt.plot(x1, y1)
    plt.xlabel('Lambda')
    plt.ylabel('Train Accuracy')
    plt.xscale('log')
    plt.savefig("part 3 train_acc plot.png")
    plt.show()

    #validation figure
    x2 = lambda_arr
    y2 = val_acc_arr
    plt.suptitle('Validation Accuracy With Different Lambda')
    plt.plot(x2, y2)
    plt.xlabel('Lambda')
    plt.ylabel('Validation Accuracy')
    plt.xscale('log')
    plt.savefig("part 3 val_acc plot.png")
    plt.show()


    return

def plot_zero(zero_weights, lambda_arr):
    x = lambda_arr
    y = zero_weights
    plt.suptitle('Zero Weights With Different Lambda')
    plt.plot(x, y)
    plt.xlabel('Lambda')
    plt.ylabel('Zero weights')
    plt.xscale('log')
    plt.savefig("part 3 zero_weights plot.png")
    plt.show()

    return

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def threshold(predict, true):
    predict = np.where(predict >= 0.5, 1, 0)

    return (predict == true).mean()


# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
# Your code here:
train_path = './IA2-train.csv'  #for train_data
train_data = load_data(train_path)
train_data = preprocess_data(train_data)

val_path = './IA2-dev.csv'    #for val_data
val_data = load_data(val_path)
val_data = preprocess_data(val_data)


#loaded_data.to_csv('test.csv')

# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambdas
# Your code here:

#(a)
# lambda_arr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# train_acc_arr = []
# val_acc_arr = []

# for lambd in lambda_arr:
#     weights, train_acc, val_acc = LR_L2_train(train_data.drop('Response', axis=1), train_data['Response'], val_data.drop('Response', axis=1), val_data['Response'], lambd)
#     train_acc_arr.append(train_acc)
#     val_acc_arr.append(val_acc)

# # print('train_acc_arr ', train_acc_arr)
# # print('val_acc_arr ', val_acc_arr)

# plot_losses(train_acc_arr, val_acc_arr, lambda_arr)

#(b)
# lambda_arr = [1e-4, 1e-2, 1]
# weights_arr = []
# top_5_weight = []

# for lambd in lambda_arr:
#     weights, train_acc, val_acc = LR_L2_train(train_data.drop('Response', axis=1), train_data['Response'], val_data.drop('Response', axis=1), val_data['Response'], lambd)
#     weights_arr.append(weights)

# for i in range(len(weights_arr)):
#     for j in np.argpartition(-1 * np.abs(weights_arr[i]), 6)[:6]:
#         if j == 0:
#             continue
#         top_5_feature_names = train_data.columns[j]
#         print("{}={}".format(top_5_feature_names,weights_arr[i][j]))
#     print('-------------------------------------------------------------------')


# print(weights_arr[0])
# print(weights_arr[0][dummy])
# print(len(weights_arr[0]))

#(c)
# lambda_arr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# weights_arr = []
# zero_weights = []

# for lambd in lambda_arr:
#     weights, train_acc, val_acc = LR_L2_train(train_data.drop('Response', axis=1), train_data['Response'], val_data.drop('Response', axis=1), val_data['Response'], lambd)
#     weights_arr.append(weights)

# weights_arr = np.array(weights_arr)

# # print(type(weights_arr))
# # print(weights_arr[0])
# # print(weights_arr[0][0])

# for i in range(len(weights_arr)):
#     zero_weights.append(np.count_nonzero(weights_arr[i] <= 1e-6))

# plot_zero(zero_weights,lambda_arr)

# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:


# Part 3  Implement logistic regression with L1 regularization and experiment with different lambdas
# Your code here:

#(a)
# lambda_arr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# train_acc_arr = []
# val_acc_arr = []

# for lambd in lambda_arr:
#     weights, train_acc, val_acc = LR_L1_train(train_data.drop('Response', axis=1), train_data['Response'], val_data.drop('Response', axis=1), val_data['Response'], lambd)
#     train_acc_arr.append(train_acc)
#     val_acc_arr.append(val_acc)

# print('train_acc_arr ', train_acc_arr)
# print('val_acc_arr ', val_acc_arr)

# plot_losses(train_acc_arr, val_acc_arr, lambda_arr)

#(b)
# lambda_arr = [1e-4, 1e-2, 1]
# weights_arr = []
# top_5_weight = []

# for lambd in lambda_arr:
#     weights, train_acc, val_acc = LR_L1_train(train_data.drop('Response', axis=1), train_data['Response'], val_data.drop('Response', axis=1), val_data['Response'], lambd)
#     weights_arr.append(weights)
#     #print("lambda ={} {}".format(lambd,weights))

# for i in range(len(weights_arr)):
#     for j in np.argpartition(-1 * np.abs(weights_arr[i]), 6)[:6]:
#         if j == 0:
#             continue
#         top_5_feature_names = train_data.columns[j]
#         print("{}={}".format(top_5_feature_names,weights_arr[i][j]))
#     print('-------------------------------------------------------------------')


#(c)
lambda_arr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
weights_arr = []
zero_weights = []

for lambd in lambda_arr:
    weights, train_acc, val_acc = LR_L1_train(train_data.drop('Response', axis=1), train_data['Response'], val_data.drop('Response', axis=1), val_data['Response'], lambd)
    weights_arr.append(weights)

weights_arr = np.array(weights_arr)

# print(type(weights_arr))
# print(weights_arr[0])
# print(weights_arr[0][0])

for i in range(len(weights_arr)):
    zero_weights.append(np.count_nonzero(weights_arr[i] <= 1e-6))

plot_zero(zero_weights,lambda_arr)

