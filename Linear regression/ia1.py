# Linear regression
# AI1 skeleton code
# By CHI-CHIEH WENG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loads a data file from a provided file location.
def load_data(path):
    loaded_data = pd.read_csv(path)
    return loaded_data

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15):
    month = []
    day = []
    year = []

    # Remove the ID column
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    # Split the date feature into three separate features
    if 'date' in data.columns:
        d = pd.to_datetime(data["date"])
        month = d.dt.month
        day = d.dt.day
        year = d.dt.year
        data.insert(0, 'month', month)
        data.insert(1, 'day', day)
        data.insert(2, 'year', year)
        data.drop('date', axis=1, inplace=True)
    # Add to the data a dummy feature
    data.insert(0, 'dummy', 1)

    # Feature age_since_renovated 
    data = data.apply(pd.to_numeric)
    for index, row in data.iterrows():
        if row['yr_renovated'] == 0:
            data.loc[index, 'age_since_renovated'] = row['year'] - row['yr_built']
        else:
            data.loc[index, 'age_since_renovated'] = row['year'] - row['yr_renovated']
    data.drop(columns=["yr_renovated"], inplace=True)

    # Normalize all features
    if normalize:
        mean = [0]*data.shape[1]
        std = [0]*data.shape[1]

        for i, col in enumerate(data.columns):
            if(col == 'waterfront' or col == 'price' or col == 'dummy'):
                continue

            mean[i] = data[col].mean()
            std[i] = data[col].std()
            data[col] = (data[col] - mean[i]) / std[i]
    
    if drop_sqrt_living15:
        data.drop('sqrt_living15', axis=1)
    return data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the pre-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr):
    weights = np.zeros(data.shape[1])
    i = 1
    losses = [np.power(data.dot(weights)-labels, 2).mean()]
    itr = 4000
    while i < itr:
        i += 1
        delta_w = data.multiply((data.dot(weights)-labels), axis=0).mean()*2
        weights = weights - lr * delta_w

        mse = round(np.power(data.dot(weights)-labels, 2).mean(), 3)
        losses.append(mse)

        if np.linalg.norm(delta_w) <= 1e-8 or mse == np.nan or mse == float("inf"):
            break
    print("itr={}, lr={}, loss={}".format(i, lr, mse))
    return weights, losses

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses):
    name = ['Learning rate 1','Learning rate 0.1','Learning rate 0.01','Learning rate 0.001','Learning rate 0.0001']
    i = 1  # 0 means lr 1 or 1 means lr 1 0.1 0.01......
    plt.suptitle('MSE & different learning rate')

    for l in losses:
        plt.plot(range(len(l)), l, label=name[i])
        plt.legend(frameon=False)
        i += 1

    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.savefig("part 1 many lr MSE plot.png")
    #plt.savefig("part 1 lr MSE plot.png")
    # plt.show()

    return

def non_plot(losses, lr):
    name = ['Learning rate 1e-14']
    i = 0

    plt.suptitle('non-normalized MSE & different learning rate')
    for l in losses:
        plt.plot(range(len(l)), l, label=name[i])
        plt.legend(frameon=False)
        i += 1

    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.savefig("non-normalized MSE 10^-14 plot.png")

    return

def compute_val_data_mse(data, labels, weights):
    return np.power(data.dot(weights) - labels, 2).mean()


# Invoke the above functions to implement the required functionality for each part of the assignment.
#--------------------------------
# Part 0  : Data preprocessing.
# Your code here:
# path = './IA1_train.csv'  #for train_data
# path = './IA1_dev.csv'    #for val_data
# loaded_data = load_data(path)
# normalize = True
# drop_sqrt_living15 = False
# loaded_data = preprocess_data(loaded_data, normalize, drop_sqrt_living15)
#--------------------------------

# Part 1 . Implement batch gradient descent and experiment with different learning rates.
###################
# process train data
train_path = './IA1_train.csv'  #for train_data
train_data = load_data(train_path)
normalize = False
drop_sqrt_living15 = False
train_data = preprocess_data(train_data, normalize, drop_sqrt_living15)


# process val data
val_path = './IA1_dev.csv'   #for val_data
val_data = load_data(val_path)
normalize = False
drop_sqrt_living15 = False
val_data = preprocess_data(val_data, normalize, drop_sqrt_living15)
###################

#(a)
#lr_arr = [1e-1, 1e-2, 1e-3, 1e-4]
#lr_arr = [1]
#losses = []

#for lr in lr_arr:
#    w, mse = gd_train(train_data.drop('price', axis=1), train_data['price'], lr)
#    print(w)
#    losses.append(mse)

#plot_losses(losses)

#(b)
#lr_arr = [1, 1e-1, 1e-2, 1e-3, 1e-4]
#weights_array = []

#for lr in lr_arr:
#    w, mse = gd_train(train_data.drop('price', axis=1), train_data['price'], lr)
#    weights_array.append(w)

#for i, w in enumerate(weights_array):
#    print("Learning rate={}, compute_val_data_MSE={}".format(lr_arr[i], compute_val_data_mse(val_data.drop("price", axis=1), val_data['price'], w)))

#(c)
#w, mse = gd_train(train_data.drop('price', axis=1), train_data['price'], 0.1)
#print("Learning rate = {}, MSE = {}".format(0.1, compute_val_data_mse(val_data.drop("price", axis=1), val_data['price'], w)))

#print(w)

#--------------------------------
# Part 2 a. Training and experimenting with non-normalized data.

#(a)
#lr_arr = [1e-14]
#losses = []

#for lr in lr_arr:
#    w, mse = gd_train(train_data.drop('price', axis=1), train_data['price'], lr)
#    losses.append(mse)
#non_plot(losses, lr)

#(b)
#lr_arr = [1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
#weights_array = []

#for lr in lr_arr:
#    w, mse = gd_train(train_data.drop('price', axis=1), train_data['price'], lr)
#    weights_array.append(w)

#for i, w in enumerate(weights_array):
#    print("Learning rate={}, compute_val_data_MSE={}".format(lr_arr[i], compute_val_data_mse(val_data.drop("price", axis=1), val_data['price'], w)))

#(c)
w, mse = gd_train(train_data.drop('price', axis=1), train_data['price'], 1e-10)
print("Learning rate = {}, MSE = {}".format(1e-10, compute_val_data_mse(val_data.drop("price", axis=1), val_data['price'], w)))
print(w)



