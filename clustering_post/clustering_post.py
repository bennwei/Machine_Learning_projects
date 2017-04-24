# coding=utf-8
# Finding related posts using KMeans clustering with NLTK and SciKit learn.
# standard dataset in machine learning is the 20newsgroup dataset, which contains 18,826 posts from 20 different newsgroups
# @ Wei Liao 2017

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.datasets
from sklearn.cluster import KMeans
import nltk.stem
import math
import scipy as sp


class StemmedCountVectorizer(CountVectorizer):
    """
        A class inherited from CountVectorizer with stemmer.
        This will do the following process for each post:
        1. The first step is lower casing the raw post in the preprocessing step
        (done in the parent class).
        2. Extracting all individual words in the tokenization step (done in the
        parent class).
        3. This concludes with converting each word into its stemmed version.
    """
    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class StemmedTfidfVectorizer(TfidfVectorizer):
    """
        A class inherited from TfidfVectorizer with stemmer.
        The resulting document vectors will not contain counts any more. Instead they will
        contain the individual TF-IDF values per term.
    """

    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def tfidf(term, doc, docset):
    """
    Calculate TD-IDF (term frequency – inverse document frequency)

    :param term:
    :param doc:
    :param docset:
    :return:

    :Example:
    >>> a, abb, abc = ["a"], ["a", "b", "b"], ["a", "b", "c"]
    >>> D = [a, abb, abc]
    >>> print(tfidf("a", a, D))
    0.0
    >>> print(tfidf("a", abb, D))
    0.0
    >>> print(tfidf("a", abc, D))
    0.0
    >>> print(tfidf("b", abb, D))
    0.270310072072
    >>> print(tfidf("a", abc, D))
    0.0
    >>> print(tfidf("b", abc, D))
    0.135155036036
    >>> print(tfidf("c", abc, D))
    0.366204096223

    """
    tf = float(doc.count(term)) / sum(doc.count(w) for w in docset)
    idf = math.log(float(len(docset))) / (len([doc for doc in docset if term in doc]))
    return tf * idf


def dist_norm(v1, v2):
    """
    Normalizing word count vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: similarity measurement
    """
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())


if __name__ == '__main__':

    # Import the 20newsgroups data
    all_data = sklearn.datasets.fetch_20newsgroups(subset='all')

    print("Total # of data is: ")
    print(len(all_data.filenames))
    print("Data filenames: ")
    print(all_data.target_names)

    # Subset a smaller data set for experimentation
    groups = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x', 'sci.space']

    # Prepare training data
    train_data = sklearn.datasets.fetch_20newsgroups(subset='train', categories=groups)

    print("# of training data")
    print(len(train_data.filenames))

    # Prepare testing data
    test_data = sklearn.datasets.fetch_20newsgroups(subset='test', categories=groups)
    print("# of testing data")
    print(len(test_data.filenames))

    # Pre-process and vectorize words
    vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                        stop_words='english', decode_error='ignore')

    vectorized = vectorizer.fit_transform(train_data.data)

    num_samples, num_features = vectorized.shape

    print("#samples: %d, #features: %d" % (num_samples, num_features))

    # Using the KMeans
    print("Start KMeans clustetring........")
    num_clusters = 50

    km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1, random_state=3)
    km.fit(vectorized)

    print(km.labels_)
    print(km.labels_.shape)

    # Testing on new post
    new_post = "Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is " \
               "working only sporadically now. I tried to format it, but now it doesn't boot any more. " \
               "Any ideas? Thanks."

    new_post_vec = vectorizer.transform([new_post])

    new_post_label = km.predict(new_post_vec)[0]
    similar_indices = (km.labels_ == new_post_label).nonzero()[0]

    # Using similar_indices, build a list of posts together with
    # their similarity scores:
    similar = []

    for i in similar_indices:
        dist = sp.linalg.norm((new_post_vec - vectorized[i].toarray()))
        similar.append((dist, train_data.data[i]))

    similar = sorted(similar)

    print("How many similar posts?")
    print(len(similar))

    # Show the top 3 similar posts
    show_at_1 = similar[0]
    show_at_2 = similar[1]
    show_at_3 = similar[2]
    # show_at_2 = similar[int(len(similar) / 10)]
    # show_at_3 = similar[int(len(similar) / 2)]

    # Show top 3 contents:
    print ("The top three contents with similarity metrics are: ")
    print(show_at_1)
    print(show_at_2)
    print(show_at_3)


















