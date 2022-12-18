# Sentiment analysis with support vector machines
# By CHI-CHIEH WENG
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loads a data file from a provided file location.
def load_data(path):
    loaded_data = pd.read_csv(path)
    return loaded_data


def preprocess_data_part0(x, y):
    # spilit trure (postive or negative)

    postive = []
    negative = []

    for i in range(len(x)):
        if x[i] == 1:
            postive.append(y[i])
        else:
            negative.append(y[i])

    #a postive
    postive_cvs = CountVectorizer(lowercase=True)
    postive_data = postive_cvs.fit_transform(postive)
    x = postive_data.toarray().sum(axis=0)

    sorted_index_array = np.argsort(-x)
    sorted_array = x[sorted_index_array]

    r = []
    times = []
    for i in sorted_array[:10]:
        r.append(postive_cvs.get_feature_names_out()[np.where(x == i)][0])
        times.append(i)

    print('Postive_CV_Top10_features', r)
    print('Postive_CV_Times',times)
    print("===========================")

    #a negative
    negative_cvs = CountVectorizer(lowercase=True)
    negative_data = negative_cvs.fit_transform(negative)
    x = negative_data.toarray().sum(axis=0)

    sorted_index_array = np.argsort(-x)
    sorted_array = x[sorted_index_array]

    r = []
    times = []
    for i in sorted_array[:10]:
        r.append(negative_cvs.get_feature_names_out()[np.where(x == i)][0])
        times.append(i)

    print('Negative_CV_Top10_features', r)
    print('Negative_CV_Times',times)
    print("===========================")


    #b postive
    postive_tfid_v = TfidfVectorizer(use_idf=True, lowercase=True)
    postive_data = postive_tfid_v.fit_transform(postive)
    x = postive_data.toarray().sum(axis=0)

    sorted_index_array = np.argsort(-x)
    sorted_array = x[sorted_index_array]

    r = []
    times = []
    for i in sorted_array[:10]:
        r.append(postive_tfid_v.get_feature_names_out()[np.where(x == i)][0])
        times.append(i)

    print('Postive_Tfid_Top10_features', r)
    print('Postive_Tfid_Times',times)
    print("===========================")

#b negative
    negative_tfid_v = TfidfVectorizer(use_idf=True, lowercase=True)
    negative_data = negative_tfid_v.fit_transform(negative)
    x = negative_data.toarray().sum(axis=0)

    sorted_index_array = np.argsort(-x)
    sorted_array = x[sorted_index_array]

    r = []
    times = []
    for i in sorted_array[:10]:
        r.append(negative_tfid_v.get_feature_names_out()[np.where(x == i)][0])
        times.append(i)

    print('Negative_Tfid_Top10_features', r)
    print('Negative_Tfid_Times',times)


def preprocess_data_part1(train_data, val_data):
    tfidf_v = TfidfVectorizer(use_idf=True, lowercase=True)
    tfidf_v.fit_transform(train_data)
    train_feature = tfidf_v.transform(train_data)
    val_feature = tfidf_v.transform(val_data)

    return train_feature, val_feature


def linearsvm(train_x, train_y, val_x, val_y, hy_c):
       
    svm_model = SVC(C = hy_c, kernel = 'linear')
    svm_model.fit(train_x, train_y)
    train_score = svm_model.score(train_x,train_y)

    val_score = svm_model.score(val_x,val_y)

    num_sup = svm_model.n_support_[0] + svm_model.n_support_[1]

    print("hyperparameter c={}, train_score:{}".format(hy_c, train_score))
    print("hyperparameter c={}, val_score:{}".format(hy_c, val_score))
    print('num_sup', num_sup)
    

    return train_score, val_score, num_sup

def quadraticsvm(train_x, train_y, val_x, val_y, hy_c):
    svm_model = SVC(C = hy_c, kernel = 'poly', degree = 2)
    svm_model.fit(train_x, train_y)
    train_score = svm_model.score(train_x,train_y)

    val_score = svm_model.score(val_x,val_y)

    num_sup = svm_model.n_support_[0] + svm_model.n_support_[1]

    print("hyperparameter c={}, train_score:{}".format(hy_c, train_score))
    print("hyperparameter c={}, val_score:{}".format(hy_c, val_score))
    print('num_sup', num_sup)

    return train_score, val_score, num_sup

def rbfsvm(train_x, train_y, val_x, val_y, hy_c, hy_g):
    svm_model = SVC(C = hy_c, kernel = 'rbf', gamma = hy_g)
    svm_model.fit(train_x, train_y)
    train_score = svm_model.score(train_x,train_y)

    val_score = svm_model.score(val_x,val_y)

    print("hyperparameter c={}, hyperparameter g={}, train_score:{}".format(hy_c, hy_g, train_score))
    print("hyperparameter c={}, hyperparameter g={}, val_score:{}".format(hy_c, hy_g, val_score))
    

    return train_score, val_score

def plot_acc(train_acc_arr, val_acc_arr, sup_arr, hyp_c):

    x = hyp_c
    y1 = train_acc_arr
    y2 = val_acc_arr
    y3 = sup_arr
    
    plt.plot(x, y1, label = 'Train_acc')
    plt.plot(x, y2, label = 'Val_acc')
    plt.xscale('log')
    plt.xlabel('Hyperparameter C')
    plt.ylabel('Score')
    plt.title('Train Accuracy and Val Accuracy With Different hyperparameter C')
    plt.legend()
    plt.savefig("Part 2.png")
    plt.show()


    plt.plot(x, y3, label = 'Support Vectors')
    plt.xscale('log')
    plt.xlabel('Hyperparameter C')
    plt.ylabel('Support Vectors')
    plt.title('Support Vectors With Different hyperparameter C')
    plt.legend()
    plt.savefig("Part 2 sup.png")
    plt.show()

def plot_heatmap_acc(train_acc_arr, val_acc_arr):

    sns.heatmap(train_acc_arr)
    plt.title('Train Accuracy hyperparameter C and Hyperparameter Gamma')
    plt.xlabel('Hyperparameter Gamma')
    plt.ylabel('Hyperparameter C')
    plt.savefig("Part 3 train_plot.png")
    plt.show()

    sns.heatmap(val_acc_arr)
    plt.title('Val Accuracy hyperparameter C and Hyperparameter Gamma')
    plt.xlabel('Hyperparameter Gamma')
    plt.ylabel('Hyperparameter C')
    plt.savefig("Part 3 val_plot.png")
    plt.show()
    

# Part 0: preprocessing
train_path = './IA3-train.csv'  #for train_data
train_data = load_data(train_path)
# train_feature = preprocess_data_part0(train_data['sentiment'],train_data['text'])

val_path = './IA3-dev.csv'    #for val_data
val_data = load_data(val_path)
# print('=====================================================================')
# print('start to val data')
# print('=====================================================================')
# val_feature = preprocess_data_part0(val_data['sentiment'],val_data['text'])
#==============================================================

#Part 1: Linear SVM
# hype_c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
# train_acc_arr = []
# val_acc_arr = []
# sup_arr = []

# train_feature, val_feature= preprocess_data_part1(train_data['text'], val_data['text'])
# print('train',train_feature.shape)
# print('val',val_feature.shape)

# for c in hype_c:
#     train_acc, val_acc, num_sup = linearsvm(train_feature, train_data['sentiment'], val_feature, val_data['sentiment'], c)
#     train_acc_arr.append(train_acc)
#     val_acc_arr.append(val_acc)
#     sup_arr.append(num_sup)

# plot_acc(train_acc_arr, val_acc_arr, sup_arr, hype_c)
#==============================================================

#Part 2. Quadratic SVM
# hype_c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
# train_acc_arr = []
# val_acc_arr = []
# sup_arr = []

# train_feature, val_feature= preprocess_data_part1(train_data['text'], val_data['text'])
# print('train',train_feature.shape)
# print('val',val_feature.shape)

# for c in hype_c:
#     train_acc, val_acc, num_sup = quadraticsvm(train_feature, train_data['sentiment'], val_feature, val_data['sentiment'], c)
#     train_acc_arr.append(train_acc)
#     val_acc_arr.append(val_acc)
#     sup_arr.append(num_sup)

# plot_acc(train_acc_arr, val_acc_arr, sup_arr, hype_c)
#==============================================================

#Part 3. SVM with RBF kernel
hype_c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
hype_g = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
# #hype_c = [1e-4, 1e-3, 1e-2]
# #hype_g = [1e-5, 1e-4, 1e-3]
# train_acc_arr = []
# val_acc_arr = []
# i = 0

# train_feature, val_feature= preprocess_data_part1(train_data['text'], val_data['text'])
# print('train',train_feature.shape)
# print('val',val_feature.shape)

# for c in hype_c:
#     train_acc_arr.append([])
#     val_acc_arr.append([])
#     for g in hype_g:
#         train_acc, val_acc = rbfsvm(train_feature, train_data['sentiment'], val_feature, val_data['sentiment'], c, g)
#         train_acc_arr[i].append(train_acc)
#         val_acc_arr[i].append(val_acc)

#     i+=1

# train_acc_plot = pd.DataFrame(train_acc_arr, index = hype_c, columns=hype_g)
# val_acc_plot = pd.DataFrame(val_acc_arr, index = hype_c, columns=hype_g)

# plot_heatmap_acc(train_acc_plot, val_acc_plot)















