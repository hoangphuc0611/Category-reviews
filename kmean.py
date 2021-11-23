from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import re
from nltk import ngrams
from underthesea import word_tokenize
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

labels_color_map = { 0: '#20b2aa', 1: '#ff7373'}

def load_data():
    df = pd.read_csv("dataM.csv")
    print(df)
    return df

# Tokenizer
def tokenizer(row):
    return word_tokenize(row, format="text")

emb = None

def embedding(X_train):
    global  emb
    emb = TfidfVectorizer(min_df=5, max_df=0.8,sublinear_tf=True)
    emb.fit(X_train)
    X_train =  emb.transform(X_train)
    # Save pkl file
    joblib.dump(emb, 'tfidf.pkl')
    return X_train

# Load data
data=load_data()

# Thực hiện tách từ
data['Text']=data.Text.apply(tokenizer)

# Chuyển các câu thành các vector sử dụng Tf-Idf
X=embedding(data['Text'])

# Khai báo thuật toán Kmean
model=KMeans(n_clusters=2,init='k-means++',n_init=10)
model.fit(X)

# Visualize data
X_visual = X.todense()
labels = model.fit_predict(X_visual)
reduced_data = PCA(n_components=2).fit_transform(X_visual)
fig, ax = plt.subplots()
for index, instance in enumerate(reduced_data):
    pca_comp_1, pca_comp_2 = reduced_data[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)

plt.show()
joblib.dump(model, 'model.pkl')