import pandas as pd
import nltk
import string
import re
import numpy as np
import pandas as pd
import pickle
#import lda

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
#from bokeh.transform import factor_cmap
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lda").setLevel(logging.WARNING)

def read_data():
    PATH = "../data/"
    train = pd.read_csv(f'{PATH}train.tsv', sep='\t')
    test = pd.read_csv(f'{PATH}test.tsv', sep='\t')

    print(train.shape)
    print(test.shape)

    print(train.dtypes)

    print(train.head())

    print(train.price.describe())

    plt.subplot(1, 2, 1)
    (train['price']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white', range=[0, 250])
    plt.xlabel('price+', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.title('Price Distribution - Training Set', fontsize=17)

    plt.subplot(1, 2, 2)
    np.log(train['price'] + 1).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('log(price+1)', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.title('Log(Price) Distribution - Training Set', fontsize=17)
    print("show")
    plt.show()
    print("show out")
    train.shipping.value_counts() / len(train)
    prc_shipBySeller = train.loc[train.shipping==1, 'price']
    prc_shipByBuyer = train.loc[train.shipping==0, 'price']
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.hist(np.log(prc_shipBySeller + 1), color='#8CB4E1', alpha=1.0, bins=50,
            label='Price when Seller pays Shipping')
    ax.hist(np.log(prc_shipByBuyer + 1), color='#007D00', alpha=0.7, bins=50,
            label='Price when Buyer pays Shipping')
    ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
    plt.xlabel('log(price+1)', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.title('Price Distribution by Shipping Type', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.show()

    print("There are %d unique values in the category column." % train['category_name'].nunique())
    print(train['category_name'].value_counts()[:5])
    print("There are %d items that do not have a label." % train['category_name'].isnull().sum())

    def split_cat(text):
        try:
            return text.split("/")
        except:
            return ("No Label", "No Label", "No Label")

    train['general_cat'], train['subcat_1'], train['subcat_2'] = \
        zip(*train['category_name'].apply(lambda x: split_cat(x)))

    test['general_cat'], test['subcat_1'], test['subcat_2'] = \
        zip(*test['category_name'].apply(lambda x: split_cat(x)))

    print("There are %d unique first sub-categories." % train['subcat_1'].nunique())
    print("There are %d unique second sub-categories." % train['subcat_2'].nunique())
    x = train['general_cat'].value_counts().index.values.astype('str')
    y = train['general_cat'].value_counts().values
    pct = [("%.2f" % (v * 100)) + "%" for v in (y / len(train))]
    trace1 = go.Bar(x=x, y=y, text=pct)
    layout = dict(title='Number of Items by Main Category',
                  yaxis=dict(title='Count'),
                  xaxis=dict(title='Category'))
    fig = dict(data=[trace1], layout=layout)
    py.iplot(fig)

    x = train['subcat_1'].value_counts().index.values.astype('str')[:15]
    y = train['subcat_2'].value_counts().values[:15]
    pct = [("%.2f" % (v * 100)) + "%" for v in (y / len(train))][:15]

    trace1 = go.Bar(x=x, y=y, text=pct,
                    marker=dict(
                        color=y, colorscale='Portland', showscale=True,
                        reversescale=False
                    ))
    layout = dict(title='Number of Items by SubCategory (Top 15)',
                  yaxis=dict(title='Count'),
                  xaxis=dict(title='SubCategory'))
    fig = dict(data=[trace1], layout=layout)
    py.iplot(fig)

    general_cats = train['general_cat'].unique()
    x = [train.loc[train['general_cat'] == cat, 'price'] for cat in general_cats]
    print(1)
    data = [go.Box(x=np.log(x[i] + 1), name=general_cats[i]) for i in range(len(general_cats))]
    layout = dict(title="Price Distribution by General Category",
                  yaxis=dict(title='Frequency'),
                  xaxis=dict(title='Category'))
    fig = dict(data=data, layout=layout)
    py.iplot(fig)
    print(2)
    print("There are %d unique brand names in the training dataset." % train['brand_name'].nunique())

    def wordCount(text):
        # convert to lower case and strip regex
        try:
             # convert to lower case and strip regex
            text = text.lower()
            regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
            txt = regex.sub(" ", text)
            # tokenize
            # words = nltk.word_tokenize(clean_txt)
            # remove words in stop words
            words = [w for w in txt.split(" ") \
                     if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]
            return len(words)
        except:
            return 0

    # add a column of word counts to both the training and test set
    train['desc_len'] = train['item_description'].apply(lambda x: wordCount(x))
    test['desc_len'] = test['item_description'].apply(lambda x: wordCount(x))

    print(train.head())

    # remove missing values in item description
    train = train[pd.notnull(train['item_description'])]


    stop = set(stopwords.words('english'))


    def tokenize(text):
        """
        sent_tokenize(): segment text into sentences
        word_tokenize(): break sentences into words
        """
        try:
            regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
            text = regex.sub(" ", text)  # remove punctuation

            tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
            tokens = []
            for token_by_sent in tokens_:
                tokens += token_by_sent
            tokens = list(filter(lambda t: t.lower() not in stop, tokens))
            filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
            filtered_tokens = [w.lower() for w in filtered_tokens if len(w) >= 3]

            return filtered_tokens

        except TypeError as e:
            print(text, e)

    # apply the tokenizer into the item descriptipn column
    train['tokens'] = train['item_description'].map(tokenize)
    test['tokens'] = test['item_description'].map(tokenize)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    for description, tokens in zip(train['item_description'].head(),
                                   train['tokens'].head()):
        print('description:', description)
        print('tokens:', tokens)
        print()

    # build dictionary with key=category and values as all the descriptions related.
    cat_desc = dict()
    for cat in general_cats:
        text = " ".join(train.loc[train['general_cat'] == cat, 'item_description'].values)
        cat_desc[cat] = tokenize(text)

    # find the most common words for the top 4 categories
    women100 = Counter(cat_desc['Women']).most_common(100)
    beauty100 = Counter(cat_desc['Beauty']).most_common(100)
    kids100 = Counter(cat_desc['Kids']).most_common(100)
    electronics100 = Counter(cat_desc['Electronics']).most_common(100)

    def generate_wordcloud(tup):
        wordcloud = WordCloud(background_color='white',
                              max_words=50, max_font_size=40,
                              random_state=42
                              ).generate(str(tup))
        return wordcloud

    fig, axes = plt.subplots(2, 2, figsize=(30, 15))

    ax = axes[0, 0]
    ax.imshow(generate_wordcloud(women100), interpolation="bilinear")
    ax.axis('off')
    ax.set_title("Women Top 100", fontsize=30)

    ax = axes[0, 1]
    ax.imshow(generate_wordcloud(beauty100))
    ax.axis('off')
    ax.set_title("Beauty Top 100", fontsize=30)

    ax = axes[1, 0]
    ax.imshow(generate_wordcloud(kids100))
    ax.axis('off')
    ax.set_title("Kids Top 100", fontsize=30)

    ax = axes[1, 1]
    ax.imshow(generate_wordcloud(electronics100))
    ax.axis('off')
    ax.set_title("Electronic Top 100", fontsize=30)

    vectorizer = TfidfVectorizer(min_df=10,
                                 max_features=180000,
                                 tokenizer=tokenize,
                                 ngram_range=(1, 2))
    all_desc = np.append(train['item_description'].values, test['item_description'].values)
    vz = vectorizer.fit_transform(list(all_desc))

    #  create a dictionary mapping the tokens to their tfidf values
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
        dict(tfidf), orient='index')
    tfidf.columns = ['tfidf']

    tfidf.sort_values(by=['tfidf'], ascending=True).head(10)

    tfidf.sort_values(by=['tfidf'], ascending=False).head(10)

    trn = train.copy()
    tst = test.copy()
    trn['is_train'] = 1
    tst['is_train'] = 0

    sample_sz = 15000

    combined_df = pd.concat([trn, tst])
    combined_sample = combined_df.sample(n=sample_sz)
    vz_sample = vectorizer.fit_transform(list(combined_sample['item_description']))


    from sklearn.decomposition import TruncatedSVD

    n_comp = 30
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_tfidf = svd.fit_transform(vz_sample)

    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)

    tsne_tfidf = tsne_model.fit_transform(svd_tfidf)

    output_notebook()
    plot_tfidf = bp.figure(plot_width=700, plot_height=600,
                           title="tf-idf clustering of the item description",
                           tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                           x_axis_type=None, y_axis_type=None, min_border=1)

    combined_sample.reset_index(inplace=True, drop=True)

    tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])
    tfidf_df['description'] = combined_sample['item_description']
    tfidf_df['tokens'] = combined_sample['tokens']
    tfidf_df['category'] = combined_sample['general_cat']

    plot_tfidf.scatter(x='x', y='y', source=tfidf_df, alpha=0.7)
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips = {"description": "@description", "tokens": "@tokens", "category": "@category"}
    show(plot_tfidf)

    from sklearn.cluster import MiniBatchKMeans

    num_clusters = 30  # need to be selected wisely
    kmeans_model = MiniBatchKMeans(n_clusters=num_clusters,
                                   init='k-means++',
                                   n_init=1,
                                   init_size=1000, batch_size=1000, verbose=0, max_iter=1000)

    kmeans = kmeans_model.fit(vz)
    kmeans_clusters = kmeans.predict(vz)
    kmeans_distances = kmeans.transform(vz)

    sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(num_clusters):
        print("Cluster %d:" % i)
        aux = ''
        for j in sorted_centroids[i, :10]:
            aux += terms[j] + ' | '
        print(aux)
        print()

    # repeat the same steps for the sample
    kmeans = kmeans_model.fit(vz_sample)
    kmeans_clusters = kmeans.predict(vz_sample)
    kmeans_distances = kmeans.transform(vz_sample)
    # reduce dimension to 2 using tsne
    tsne_kmeans = tsne_model.fit_transform(kmeans_distances)

    colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
                         "#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053",
                         "#5e9981",
                         "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce",
                         "#d07d3c",
                         "#52697d", "#194196", "#d27c88", "#36422b", "#b68f79"])

    # combined_sample.reset_index(drop=True, inplace=True)
    kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
    kmeans_df['cluster'] = kmeans_clusters
    kmeans_df['description'] = combined_sample['item_description']
    kmeans_df['category'] = combined_sample['general_cat']
    # kmeans_df['cluster']=kmeans_df.cluster.astype(str).astype('category')

    plot_kmeans = bp.figure(plot_width=700, plot_height=600,
                            title="KMeans clustering of the description",
                            tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                            x_axis_type=None, y_axis_type=None, min_border=1)

    source = ColumnDataSource(data=dict(x=kmeans_df['x'], y=kmeans_df['y'],
                                        color=colormap[kmeans_clusters],
                                        description=kmeans_df['description'],
                                        category=kmeans_df['category'],
                                        cluster=kmeans_df['cluster']))

    plot_kmeans.scatter(x='x', y='y', color='color', source=source)
    hover = plot_kmeans.select(dict(type=HoverTool))
    hover.tooltips = {"description": "@description", "category": "@category", "cluster": "@cluster"}
    show(plot_kmeans)

    cvectorizer = CountVectorizer(min_df=4,
                                  max_features=180000,
                                  tokenizer=tokenize,
                                  ngram_range=(1, 2))

    cvz = cvectorizer.fit_transform(combined_sample['item_description'])

    lda_model = LatentDirichletAllocation(n_components=20,
                                          learning_method='online',
                                          max_iter=20,
                                          random_state=42)

    X_topics = lda_model.fit_transform(cvz)

    n_top_words = 10
    topic_summaries = []

    topic_word = lda_model.components_  # get the topic words
    vocab = cvectorizer.get_feature_names()

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        topic_summaries.append(' '.join(topic_words))
        print('Topic {}: {}'.format(i, ' | '.join(topic_words)))

    tsne_lda = tsne_model.fit_transform(X_topics)

    unnormalized = np.matrix(X_topics)
    doc_topic = unnormalized / unnormalized.sum(axis=1)

    lda_keys = []
    for i, tweet in enumerate(combined_sample['item_description']):
        lda_keys += [doc_topic[i].argmax()]

    lda_df = pd.DataFrame(tsne_lda, columns=['x', 'y'])
    lda_df['description'] = combined_sample['item_description']
    lda_df['category'] = combined_sample['general_cat']
    lda_df['topic'] = lda_keys
    lda_df['topic'] = lda_df['topic'].map(int)

    plot_lda = bp.figure(plot_width=700,
                         plot_height=600,
                         title="LDA topic visualization",
                         tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    source = ColumnDataSource(data=dict(x=lda_df['x'], y=lda_df['y'],
                                        color=colormap[lda_keys],
                                        description=lda_df['description'],
                                        topic=lda_df['topic'],
                                        category=lda_df['category']))

    plot_lda.scatter(source=source, x='x', y='y', color='color')
    hover = plot_kmeans.select(dict(type=HoverTool))
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = {"description": "@description",
                      "topic": "@topic", "category": "@category"}
    show(plot_lda)

    def prepareLDAData():
        data = {
            'vocab': vocab,
            'doc_topic_dists': doc_topic,
            'doc_lengths': list(lda_df['len_docs']),
            'term_frequency':cvectorizer.vocabulary_,
            'topic_term_dists': lda_model.components_
        }
        return data

    import pyLDAvis

    lda_df['len_docs'] = combined_sample['tokens'].map(len)
    ldadata = prepareLDAData()
    pyLDAvis.enable_notebook()
    prepared_data = pyLDAvis.prepare(**ldadata)

    import IPython.display
    from IPython.core.display import display, HTML, Javascript

    # h = IPython.display.display(HTML(html_string))
    # IPython.display.display_HTML(h)
if __name__ == '__main__':
    read_data()
