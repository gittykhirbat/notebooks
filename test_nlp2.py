from itertools import chain

import numpy as np

from jean.nlp import embeddings
from jean.nlp2 import embeddings as embeddings2

docs = [
    '''
    Far out in the uncharted backwaters of the unfashionable end of
    the western spiral arm of the Galaxy lies a small unregarded
    yellow sun.
    ''',
    '''
    Orbiting this at a distance of roughly ninety-two million miles
    is an utterly insignificant little blue green planet whose ape-
    descended life forms are so amazingly primitive that they still
    think digital watches are a pretty neat idea.
    ''',
    '''
    This planet has - or rather had - a problem, which was this: most
    of the people living on it were unhappy for pretty much of the time.
    Many solutions were suggested for this problem, but most of these
    were largely concerned with the movements of small green pieces
    of paper, which is odd because on the whole it wasn't the small
    green pieces of paper that were unhappy.
    '''
]


def test_embed_text():
    x1 = np.array([phrase.vector.flatten() for phrase in embeddings.embed_phrases(docs[:1])]).reshape(1, -1)
    x2 = embeddings2.embed_text(docs[0])

    assert x2.shape == (1, 700)
    assert np.array_equal(x1, x2)


def test_embed_text_many():
    x1 = np.array([phrase.vector.flatten() for phrase in embeddings.embed_phrases(docs)]).reshape(len(docs), -1)
    x2 = embeddings2.embed_text(docs)

    assert x2.shape == (len(docs), 700)
    assert np.array_equal(x1, x2)


def test_embed_document():
    x1 = embeddings.embed_document(docs[0]).vector
    x2 = embeddings2.embed_document(docs[0])

    assert x2.shape == (1, 700)
    assert np.array_equal(x1, x2)


def test_embed_documents():
    x1 = np.array([embeddings.embed_document(doc).vector.flatten() for doc in docs])
    x2 = embeddings2.embed_documents(docs)

    assert x2.shape == (len(docs), 700)
    assert np.array_equal(x1, x2)


def test_pos_tag_text():
    x1 = list(chain(*embeddings.pos_tag_raw_text(docs[0])))
    x2 = embeddings2.pos_tag_text(docs[0])

    assert x1 == x2


def test_pos_tag_text_many():
    x1 = [list(chain(*embeddings.pos_tag_raw_text(doc))) for doc in docs]
    x2 = embeddings2.pos_tag_text(docs)

    assert x1 == x2
