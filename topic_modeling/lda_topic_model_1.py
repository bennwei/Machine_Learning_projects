# coding=utf-8
# Topic modeling of collection of news reports from the Associated Press (AP) using gensim package
# @ Wei Liao 2017

from __future__ import print_function
from wordcloud import create_cloud

try:
    from gensim import corpora, models, matutils
except:
    print("Import gensim falied!")
    print()
    print("Please install it!")
    raise

import matplotlib.pyplot as plt
import numpy as np
from os import path

num_topics = 100

# Check data exists
if not path.exists('./data/ap/ap.dat'):
    print('Error: Expected data to be present at data/ap/')
    print('Please cd into ./data & run ./download_ap.sh')

# Load the data
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

# Build topic model
model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=corpus.id2word, alpha=None)

# Iterate over all the topics in the model
for i in range(model.num_topics):
    words = model.show_topic(i, 64)
    tf = sum(f for f, w in words)
    with open('topics.txt', 'w') as output:
        output.write('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for f, w in words))
        output.write("\n\n\n")

# identify the most discussed topic, i.e., the one with the highest total weight

topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
weight = topics.sum(1)
max_topic = weight.argmax()

# Get the top 64 words for this topic, Without the argument, show_topic would return only 10 words

words = model.show_topic(max_topic, 64)

# This function will actually check for the presence of pytagcloud and is otherwise a non-operation

create_cloud('cloud_blei_lda.png', words)

num_topics_used = [len(model[doc]) for doc in corpus]
fig, ax = plt.subplot()
ax.hist(num_topics_used, np.arange(42))
ax.set_ylabel('Number of documents')
ax.set_xlabel('Number of topics')
fig.tight_layout()
fig.savefig('Figure_04_01.png')

# repeat the same exercise using alpha=1.0, or play around with this parameter

ALPHA = 1.0

model1 = models.ldamodel.LdaModel(
    corpus, num_topics=num_topics, id2word=corpus.id2word, alpha=ALPHA)
num_topics_used1 = [len(model1[doc]) for doc in corpus]

fig,ax = plt.subplots()
ax.hist([num_topics_used, num_topics_used1], np.arange(42))
ax.set_ylabel('Number of documents')
ax.set_xlabel('Number of topics')

# The coordinates below were fit by trial and error to look good
ax.text(9, 223, r'default alpha')
ax.text(26, 156, 'alpha=1.0')
fig.tight_layout()
fig.savefig('Figure_04_02.png')



