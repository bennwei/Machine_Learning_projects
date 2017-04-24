# coding=utf-8
# Topic modeling of Wikipedia dump using LDA with gensim package
# @ Wei Liao 2017

from __future__ import print_function
import logging
import gensim
import numpy as np

NUM_OF_TOPICS = 100

# Set up logging in order to get progress information as the model is being built:

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load preprocessed corpus:
id2words = gensim.corpora.Dictionary.load_from_text('data/wiki_en_output_wordids.txt.bz2')
mm = gensim.corpora.MmCorpus('data/wiki_en_output_tfidf.mm')

# Calling the constructor is enough to build the model
# This call will take a few hours!

model = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2words, num_topics=NUM_OF_TOPICS,
                                        update_every=1, chunksize=10000, passes=1)

# Or use HDP model
# model = gensim.models.hdpmodel.HdpModel(corpus=mm, id2word=id2words, chunksize=10000)

# Save the model for future use

model.save('wiki_lda.pkl')

# Compute the document/topic matrix
topics = np.zeros((len(mm), model.num_topics))
for di, doc in enumerate(mm):
    doc_top = model[doc]
    for ti, tv in doc_top:
        topics[di, ti] += tv
np.save('topic.npy', topics)

# Alternatively, we create a sparse matrix and save that. This alternative
# saves disk space, at the cost of slightly more complex code:

## from scipy import sparse, io
## sp = sparse.csr_matrix(topics)
## io.savemat('topics.mat', {'topics': sp})






