"""TF-IDF latent semantic indexing of a cell-by-feature count matrix."""

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def tfidf_lsi(matrix, n_components=32):
    """L2-normalized LSI coordinates, used as an auxiliary space when searching anchors."""
    tf = matrix.copy().astype(np.float32)
    rows = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
    rows[rows <= 0] = 1
    tf = sp.diags(1.0 / rows) @ tf
    binary = matrix.copy().astype(np.float32)
    binary.data[:] = 1
    df = np.asarray(binary.sum(axis=0)).ravel().astype(np.float32)
    df[df <= 0] = 1
    idf = (float(matrix.shape[0]) / df).astype(np.float32)
    tfidf = tf @ sp.diags(idf)
    rows = np.asarray(tfidf.sum(axis=1)).ravel().astype(np.float32)
    rows[rows <= 0] = 1
    tfidf = sp.diags(1e4 / rows) @ tfidf
    tfidf.data = np.log1p(tfidf.data).astype(np.float32)
    svd = TruncatedSVD(
        n_components=min(n_components, max(1, min(tfidf.shape) - 1)),
        algorithm="randomized",
        n_iter=5,
        random_state=0,
        tol=0.0,
    )
    return normalize(svd.fit_transform(tfidf), norm="l2").astype(np.float32)
