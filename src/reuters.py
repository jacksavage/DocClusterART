# cluster the reuters dataset
# 1) preprocess and tokenize with spacy
# 2) vectorize each doc by training a doc2vec nnet with gensim
# 3) cluster these vectors using Fuzzy ART (todo and TopoART)
# 4) compare document categories to their labels

import gensim, spacy, nltk, os, numpy, fuzzy_art, init
import logging as log

config = init.run()

# point to the dataset
nltk.data.path.append(config["nltk_path"])
loader = nltk.corpus.reuters
reuters_data = os.path.join("data", "reuters")

# store file IDs to ensure consistent ordering
fname_fileids = os.path.join(reuters_data, "fileids")
if not os.path.isfile(fname_fileids):
    with open(fname_fileids, "w") as file:
        file.writelines(fileid + "\n" for fileid in loader.fileids())

with open(fname_fileids, "r") as file:
    fileids = [fileid for fileid in file]

# tokenize each document
fname_doclines = os.path.join(reuters_data, "doclines")
if not os.path.isfile(fname_doclines):
    nlp = spacy.load("en_core_web_sm")
    is_token_allowed = lambda token : not token.is_stop and len(token) > 2 and token.is_alpha
    to_wordlist = lambda doc : [token.lemma_.lower() for token in nlp(doc) if is_token_allowed(token)]

    log.info(f"Generating doclines file at {fname_doclines}")
    corpus = (loader.raw(file_id) for file_id in fileids)
    with open(fname_doclines, "w") as file:
        file.writelines(" ".join(to_wordlist(doc)) + "\n" for doc in corpus)

with open(fname_doclines, "r") as doclines:
    log.info(f"Loading doclines file from {fname_doclines}")
    # tag each document with an ID
    docs = (dl.split() for i, dl in enumerate(doclines) if fileids[i].startswith("train"))
    tokenized_corpus = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]

# load or train the doc to vec model
fname_docvec = os.path.join(reuters_data, "docvec")
vector_size = 50
if os.path.isfile(fname_docvec):
    log.info(f"Loading doc2vec model from {fname_docvec}")
    model = gensim.models.doc2vec.Doc2Vec.load(fname_docvec)
else:
    log.info("Training document vectorizer")
    model = \
        gensim.models.doc2vec.Doc2Vec(
            tokenized_corpus, 
            dm = 1,
            vector_size=vector_size,
            window=2,
            workers=8,
            dm_concat=1,
            min_count=1
        )
    log.info(f"Saving doc2vec model to {fname_docvec}")
    model.save(fname_docvec)

# cluster the documents with fuzzy ART
# classify documents from the test set
fa = fuzzy_art.FuzzyArt(vector_size, vigilance=0.7, choice=0.001, learn_rate=1.0)
y = numpy.zeros(len(fileids), int)
with open(fname_doclines, "r") as doclines:
    log.info("Training fuzzy ART")
    docs = ((i, dl.split()) for i, dl in enumerate(doclines) if fileids[i].startswith("train"))
    for i, doc in docs:
        x = model.infer_vector(doc)
        # assert x.min() >= 0.0 and x.max() <= 1.0
        # x = fuzzy_art.complement_code(x)
        y[i] = fa.train(x)

    log.info("Evaluating test docs with fuzzy ART")
    doclines.seek(0)
    docs = ((i, dl.split()) for i, dl in enumerate(doclines) if fileids[i].startswith("test"))
    for i, doc in docs:
        x = model.infer_vector(doc)
        # assert x.min() >= 0.0 and x.max() <= 1.0
        # x = fuzzy_art.complement_code(x)
        y[i] = fa.choose_category(x)

# save the results
fname_labels = os.path.join(reuters_data, "labels")
log.info(f"Writing labels to {fname_labels}")
numpy.savetxt(fname_labels, y, fmt="%d")

# save the category weights
fname_facats = os.path.join(reuters_data, "facats")
log.info(f"Writing fuzzy ART category weights to {fname_facats}")
w = numpy.stack(fa.categories, axis=0)
numpy.savetxt(fname_facats, w)

# save the category counts
fname_facnts = os.path.join(reuters_data, "facnts")
log.info(f"Writing fuzzy ART category counts to {fname_facnts}")
numpy.savetxt(fname_facnts, fa.category_counts, fmt="%d")
