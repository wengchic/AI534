import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier


# Loads GloVe embeddings from a designated file location. 
#
# Invoked via:
# ge = GloVe_Embedder(path_to_embeddings)
#
# Embed single word via:
# embed = ge.embed_str(word)
#
# Embed a list of words via:
# embeds = ge.embed_list(word_list)
#
# Find nearest neighbors via:
# ge.find_k_nearest(word, k)
#
# Save vocabulary to file via:
# ge.save_to_file(path_to_file)

class GloVe_Embedder:
    def __init__(self, path):
        self.embedding_dict = {}
        self.embedding_array = []
        self.unk_emb = 0
        # Adapted from https://stackoverflow.com/questions/37793118/load-pretrained-GloVe-vectors-in-python
        with open(path,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.embedding_dict[word] = embedding
                self.embedding_array.append(embedding.tolist())
        self.embedding_array = np.array(self.embedding_array)
        self.embedding_dim = len(self.embedding_array[0])
        self.vocab_size = len(self.embedding_array)
        self.unk_emb = np.zeros(self.embedding_dim)

    # Check if the provided embedding is the unknown embedding.
    def is_unk_embed(self, embed):
        return np.sum((embed - self.unk_emb) ** 2) < 1e-7
    
    # Check if the provided string is in the vocabulary.
    def token_in_vocab(self, x):
        if x in self.embedding_dict and not self.is_unk_embed(self.embedding_dict[x]):
            return True
        return False

    # Returns the embedding for a single string and prints a warning if
    # the string is unknown to the vocabulary.
    # 
    # If indicate_unk is set to True, the return type will be a tuple of 
    # (numpy array, bool) with the bool indicating whether the returned 
    # embedding is the unknown embedding.
    #
    # If warn_unk is set to False, the method will no longer print warnings
    # when used on unknown strings.
    def embed_str(self, x, indicate_unk = False, warn_unk = False):
        if self.token_in_vocab(x):
            if indicate_unk:
                return (self.embedding_dict[x], False)
            else:
                return self.embedding_dict[x]
        else:
            if warn_unk:
                    print("Warning: provided word is not part of the vocabulary!")
            if indicate_unk:
                return (self.unk_emb, True)
            else:
                return self.unk_emb

    # Returns an array containing the embeddings of each vocabulary token in the provided list.
    #
    # If include_unk is set to False, the returned list will not include any unknown embeddings.
    def embed_list(self, x, include_unk = True):
        if include_unk:
            embeds = [self.embed_str(word, warn_unk = False).tolist() for word in x]
        else:
            embeds_with_unk = [self.embed_str(word, indicate_unk=True, warn_unk = False) for word in x]
            embeds = [e[0].tolist() for e in embeds_with_unk if not e[1]]
            if len(embeds) == 0:
                print("No known words in input:" + str(x))
                embeds = [self.unk_emb.tolist()]
        return np.array(embeds)
    
    # Finds the vocab words associated with the k nearest embeddings of the provided word. 
    # Can also accept an embedding vector in place of a string word.
    # Return type is a nested list where each entry is a word in the vocab followed by its 
    # distance from whatever word was provided as an argument.
    def find_k_nearest(self, word, k, warn_about_unks = True):
        if type(word) == str:
            word_embedding, is_unk = self.embed_str(word, indicate_unk = True)
        else:
            word_embedding = word
            is_unk = False
        if is_unk and warn_about_unks:
            print("Warning: provided word is not part of the vocabulary!")

        all_distances = np.sum((self.embedding_array - word_embedding) ** 2, axis = 1) ** 0.5
        distance_vocab_index = [[w, round(d, 5)] for w,d,i in zip(self.embedding_dict.keys(), all_distances, range(len(all_distances)))]
        distance_vocab_index = sorted(distance_vocab_index, key = lambda x: x[1], reverse = False)
        return distance_vocab_index[:k]

    def save_to_file(self, path):
        with open(path, 'w') as f:
            for k in self.embedding_dict.keys():
                embedding_str = " ".join([str(round(s, 5)) for s in self.embedding_dict[k].tolist()])
                string = k + " " + embedding_str
                f.write(string + "\n")


def load_data(path):
    loaded_data = pd.read_csv(path)
    return loaded_data

def create_y_true(voc_list):
    y_true = np.zeros(150)
    i = 0
    c = 0

    while c < 150:
        y_true[c] = i
        c += 1

        if c == 30 or c == 60 or c == 90 or c == 120:
            i += 1
    
    return y_true.astype(int)

def purity(y_true, y_pred):
    matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

def split_word(data):
    tweet_sents = []
    tweet_words = []

    for sent in tqdm(data):
        sent = re.sub(r"http\S+", "", sent)
        sent = re.sub("\S*\d\S*", "", sent).strip()
        sent = re.sub('[^A-Za-z]+', ' ', sent)
        sent = ' '.join(e.lower() for e in sent.split() if e.lower() not in stop)
        tweet_sents.append(sent.strip())

    for words in tweet_sents:
        tweet_words.append(words.split())

    return tweet_sents, tweet_words

def TfidfV(train_data, val_data):
    tfidf_v = TfidfVectorizer(use_idf=True, lowercase=True)
    tfidf_v.fit_transform(train_data)

    #train_feature = tfidf_v.transform(train_data)
    #val_feature = tfidf_v.transform(val_data)
    feature_name = tfidf_v.get_feature_names()
    dictionary = dict(zip(tfidf_v.get_feature_names(), tfidf_v.idf_))

    return dictionary, feature_name

def cal_vector(spilt_word, feature_name, dictionary):
    total_vec = []

    for tweet_words in tqdm(spilt_word):
        tem_vec = np.zeros(200)
        weight_sum = 0

        for word in tweet_words:
            if word in feature_name:
                embed_vec = ge.embed_str(word)

                tf_idf_vec = dictionary[word] * (tweet_words.count(word)/len(tweet_words))
                tem_vec += (embed_vec * tf_idf_vec)
                weight_sum += tf_idf_vec

        if weight_sum != 0:
            tem_vec /= weight_sum

        total_vec.append(tem_vec)

    return total_vec

def cal_vector_test(spilt_word, feature_name, dictionary):
    total_vec = []

    for tweet_words in tqdm(spilt_word):
        tem_vec = np.zeros(200)
        avg_weight = 0
        weight_sum = 0

        for word in tweet_words:
            if word in feature_name:
                embed_vec = ge.embed_str(word)
                tf_idf_vec = dictionary[word]
                tem_vec += (embed_vec * tf_idf_vec)
                weight_sum += tf_idf_vec
                
            avg_weight = (tem_vec/abs(weight_sum))

        total_vec.append(avg_weight)

    return total_vec

def linearsvm(train_x, train_y, val_x, val_y, hy_c):
       
    svm_model = SVC(C = hy_c, kernel = 'linear')
    svm_model.fit(train_x, train_y)

    train_score = svm_model.score(train_x, train_y)
    val_score = svm_model.score(val_x, val_y)

    print("hyperparameter c={}, train_score:{}".format(hy_c, train_score))
    print("hyperparameter c={}, val_score:{}".format(hy_c, val_score))

    return train_score, val_score

def RF(train_x, train_y, val_x, val_y, hy_n):
    rf_model = RandomForestClassifier(n_estimators = hy_n)
    rf_model.fit(train_x, train_y)

    train_score = rf_model.score(train_x, train_y)
    val_score = rf_model.score(val_x, val_y)

    print("hyperparameter n={}, train_score:{}".format(hy_n, train_score))
    print("hyperparameter n={}, val_score:{}".format(hy_n, val_score))

    return train_score, val_score

def plot_kms(k_scores, clusts):
    x = clusts
    y = k_scores

    plt.plot(x, y, label = 'Kmeans Score')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Kmeans Score')
    plt.title('Kmeans Score With Different Cluster')
    plt.legend()
    plt.savefig("Kmeans Score.png")
    plt.show()

def plot_p_score(p_scores, clusts):
    x = clusts
    y = p_scores

    plt.plot(x, y, label = 'Purity Score')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Purity Score')
    plt.title('Purity Score With Different Cluster')
    plt.legend()
    plt.savefig("Purity.png")
    plt.show()

def plot_r_score(r_scores, clusts):
    x = clusts
    y = r_scores

    plt.plot(x, y, label = 'Rand Score')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Rand Score')
    plt.title('Rand Score With Different Cluster')
    plt.legend()
    plt.savefig("Rand.png")
    plt.show()

def plot_n_score(n_scores, clusts):
    x = clusts
    y = n_scores

    plt.plot(x, y, label = 'Mutual Score')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Mutual Score')
    plt.title('Mutual Score With Different Cluster')
    plt.legend()
    plt.savefig("Mutual.png")
    plt.show()

def plot_acc(train_acc_arr, val_acc_arr, hyp_c):

    x = hyp_c
    y1 = train_acc_arr
    y2 = val_acc_arr
    
    plt.plot(x, y1, label = 'Train_acc')
    plt.plot(x, y2, label = 'Val_acc')
    plt.xscale('log')
    plt.xlabel('Hyperparameter C')
    plt.ylabel('Score')
    plt.title('Train Accuracy and Val Accuracy With Different hyperparameter C')
    plt.legend()
    plt.savefig("first way.png")
    plt.show()


# Part 1 Explore word embeddings
# (a) Build your own data set of words
# emb_path = './GloVe_Embedder_data.txt'
# word_list = ['flight', 'good', 'terrible', 'help', 'late']
# #word = 'flight'

# ge = GloVe_Embedder(emb_path)

# for word in word_list:
#     print('word :', word)
#     print('\n')
#     print(ge.find_k_nearest(word, 30))
#     print('=========================================')
#=================================================================
# (b) Dimension reduction and visualization.
# (1)
# emb_path = './GloVe_Embedder_data.txt'
# word_list = ['flight', 'good', 'terrible', 'help', 'late']
# find_voc = []
# find_k = []
# ge = GloVe_Embedder(emb_path)

# embeds = ge.embed_list(word_list)
# #print(embeds[0].shape)
## find 29 similar words
# for word in embeds:
#     l = ge.find_k_nearest(word, 30)

#     for i in range(len(l)):
#         find_voc.append(l[i][0])

# #print(len(find_voc))

# #k = ge.embed_list(find_voc)
# #print(type(k))


# pca = PCA(n_components=2)
# X = pca.fit_transform(ge.embed_list(find_voc))
# # X = X.reshape(5,30,2)
# # print(X.shape)
# #plt.scatter(X[:,0], X[:,1], c = colors, cmap = plt.cm.Spectral)
# plt.title('PCA Plot')
# plt.scatter(X[0:30 ,0],X[0:30 ,1], c = "r", label = 'flight')
# plt.scatter(X[30:60 ,0],X[30:60 ,1], c = "g", label = 'good')
# plt.scatter(X[60:90 ,0],X[60:90 ,1], c = "b", label = 'terrible')
# plt.scatter(X[90:120 ,0],X[90:120 ,1], c = "k", label = 'help')
# plt.scatter(X[120:150 ,0],X[120:150 ,1], c = "y", label = 'late')
# plt.legend()
# plt.savefig("PCA.png")
# plt.show()
#=================================================================
# (2)
# emb_path = './GloVe_Embedder_data.txt'
# word_list = ['flight', 'good', 'terrible', 'help', 'late']
# find_voc = []
# find_k = []

# ge = GloVe_Embedder(emb_path)
# embeds = ge.embed_list(word_list)

# for word in embeds:
#     l = ge.find_k_nearest(word, 30)

#     for i in range(len(l)):
#         find_voc.append(l[i][0])

# tsne = TSNE(n_components=2, perplexity = 50) #perplexity = 5, 10, 30, 50
# X = tsne.fit_transform(ge.embed_list(find_voc))

# plt.title('TSNE with Perplexity 50')
# plt.scatter(X[0:30 ,0],X[0:30 ,1], c = "r", label = 'flight')
# plt.scatter(X[30:60 ,0],X[30:60 ,1], c = "g", label = 'good')
# plt.scatter(X[60:90 ,0],X[60:90 ,1], c = "b", label = 'terrible')
# plt.scatter(X[90:120 ,0],X[90:120 ,1], c = "k", label = 'help')
# plt.scatter(X[120:150 ,0],X[120:150 ,1], c = "y", label = 'late')
# plt.legend()
# plt.savefig("TSNE 50.png")
# plt.show()
#========================================================================
# (c) Clustering
# (1)
# emb_path = './GloVe_Embedder_data.txt'
# word_list = ['flight', 'good', 'terrible', 'help', 'late']
# clusts = [2, 5, 10, 15, 20]
# k_scores = []
# find_voc = []
# find_k = []

# ge = GloVe_Embedder(emb_path)
# embeds = ge.embed_list(word_list)

# for word in embeds:
#     l = ge.find_k_nearest(word, 30)

#     for i in range(len(l)):
#         find_voc.append(l[i][0])

# find_voc = ge.embed_list(find_voc)

# for c in clusts:
#     kms = KMeans(n_clusters = c).fit(find_voc)
#     k_score = kms.inertia_
#     #print(k_score)
#     k_scores.append(k_score)
    

# plot_kms(k_scores, clusts)
#========================================================================
# (2)
# emb_path = './GloVe_Embedder_data.txt'
# word_list = ['flight', 'good', 'terrible', 'help', 'late']
# clusts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# clusts = [2, 3, 5]
# k_scores = []
# p_scores = []
# r_scores = []
# n_scores = []
# find_voc = []
# find_k = []

# ge = GloVe_Embedder(emb_path)
# embeds = ge.embed_list(word_list)

# for word in embeds:
#     l = ge.find_k_nearest(word, 30)

#     for i in range(len(l)):
#         find_voc.append(l[i][0])


# find_voc = ge.embed_list(find_voc)

# for c in clusts:
#     kms = KMeans(n_clusters = c).fit(find_voc)
#     #y_predict = kms.predict(find_voc)
#     y_predict = kms.labels_

#     y_true = create_y_true()

#     print('y_predict', y_predict)
#     print('y_ture', y_true)

#     p_score = purity(y_true, y_predict)
#     p_scores.append(p_score)

#     r_score = adjusted_rand_score(y_true, y_predict)
#     r_scores.append(r_score)

#     n_score = normalized_mutual_info_score(y_true, y_predict)
#     n_scores.append(n_score)

# print('p_scores', p_scores)
# print('r_scores', r_scores)
# print('n_scores', n_scores)

# plot_p_score(p_scores, clusts)
# plot_r_score(r_scores, clusts)
# plot_n_score(n_scores, clusts)
#========================================================================
# Part 2 Using word embeddings to improve classification
hype_c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
#hype_c = [1e-2, 1e-1, 1, 10, 100]
train_tweet = []
val_tweet = []
train_words = []
val_words = []
train_emd_vectors = []
val_emd_vectors = []
train_acc_arr = []
val_acc_arr = []

# remove words
stop= set(['the', 'I', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'im', 'ewr', \
            'u', 'x', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'aa', 'will', 'just', 'don', "didn", 've', "y", 'now', 'd', 'll', 'm', 'o', 're', \
            'wasn', 'shan', 'ain', 'aren', "dc", 'couldn', "rep", 'didn', "wasn", 'doesn', "weren", 'hadn',\
            "needn", 'hasn', "needn", 'shouldn', "mustn't", 'isn', "wsj", 'ma', 'mightn', "b", 'mustn', 'f', 'w', 'nc', 'ugh', 'ada', 'r', 'den', 'oh',\
            'phl', 'clt', 'erie', 'ua', 'sjc', 'nyc', 'sir', 'dm', 'fri', 'sit'])

train_path = './IA3-train.csv'  #for train_data
train_data = load_data(train_path)

val_path = './IA3-dev.csv'    #for val_data
val_data = load_data(val_path)

emb_path = './GloVe_Embedder_data.txt'
ge = GloVe_Embedder(emb_path)


# split words
train_tweet, train_words = split_word(train_data['text'].values)
val_tweet, val_words = split_word(val_data['text'].values)

# create each words weight
dictionary, feature_name = TfidfV(train_tweet, val_tweet)

# cal vect (each word embed * weights)/sum(weights)
train_emd_vectors = cal_vector_test(train_words, feature_name, dictionary)

#print(train_emd_vectors)
val_emd_vectors = cal_vector_test(val_words, feature_name, dictionary)

for c in tqdm(hype_c):
    train_acc, val_acc = linearsvm(train_emd_vectors, train_data['sentiment'], val_emd_vectors, val_data['sentiment'], c)

    train_acc_arr.append(train_acc)
    val_acc_arr.append(val_acc)

plot_acc(train_acc_arr, val_acc_arr, hype_c)










