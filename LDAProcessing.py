import nltk
import re

stop_list = nltk.corpus.stopwords.words('english')
# The following list is to further remove some frequent words in SGNews.
#stop_list += ['apple', 'samsung', 'archos', 'creative', 'iriver', 'rio', 'sandisk', 'sony', 'zune']

import gensim

def load_corpus( dir ):
    # dir is a directory with plain text files to load.
    corpus = nltk.corpus.PlaintextCorpusReader(dir, '.*')
    return corpus

def corpus2docs ( corpus ):
    # corpus is a object returned by load_corpus that represents a corpus.
    fids = corpus.fileids()
    docs1 = []
    for fid in fids:
        doc_raw = corpus.raw(fid)
        doc = nltk.word_tokenize(doc_raw)
        docs1.append(doc)
    return docs1

def docs2vecs ( docs , dictionary ):
    
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1

    
