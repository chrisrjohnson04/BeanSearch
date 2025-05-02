from genericpath import isfile
from posixpath import isdir
from sys import int_info
from typing import List, Tuple, Dict
from collections.abc import Callable
import math
from collections import defaultdict
from collections import Counter
import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import numpy as np
import nltk
from nltk.stem import PorterStemmer


# init stemmer
stemmer = PorterStemmer()


def tokenize(text: str) -> List[str]:
    """Returns a list of words that make up the text.
    Words are stemmed using Porter stemmer.

    Parameters
    ----------
    text : str
        The input string to be tokenized.

    Returns
    -------
    List[str]
        A list of stemmed strings representing the words in the text.
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    # Apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


def tokenize_beans(input_coffee) -> List[List[str]]:
    """Returns a list of tokens for every bean

    Parameters
    ----------
    input_coffee : json object

    Returns
    -------
    List[List[str]]
        A list of tokens for every combined bean description
    """
    tokens = []
    combined_descriptions = [
        bean["desc_1"] + bean["desc_2"] + bean["desc_3"] for bean in input_coffee]
    for desc in combined_descriptions:
        tokens.append(tokenize(desc))
    return tokens


def build_inverted_index(beans: List[List[str]]) -> dict:
    """Builds an inverted index from the messages.

    Arguments
    =========

    beans: list of lists.
        Each list contains the tokens for a bean's combined description

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]
    """
    ii = {}

    for doc_id, bean_tokens in enumerate(beans):
        freq = {}
        for token in bean_tokens:
            freq[token] = freq.get(token, 0) + 1

        for token, count in freq.items():
            if token not in ii:
                ii[token] = [(doc_id, count)]
            else:
                ii[token].append((doc_id, count))

    # sort by doc_id
    for token in ii:
        ii[token] = sorted(ii[token], key=lambda x: x[0])

    return ii


def compute_idf(inv_idx, n_docs, min_df=1, max_df_ratio=0.9):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Parameters
    ==========

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """

    res = {}
    for word, postings in inv_idx.items():
        df = len(postings)
        if df < min_df:
            continue
        if df > max_df_ratio * n_docs:
            continue
        res[word] = math.log(n_docs / (1 + df), 2)
    return res


def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.

    Arguments
    =========

    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.

    Returns
    =======

    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """

    res = [0.0] * n_docs

    for term, postings in index.items():
        if term in idf:
            idf_i = idf[term]
            for doc_id, tf in postings:
                res[doc_id] += (tf * idf_i) ** 2

    res = [math.sqrt(val) for val in res]
    return res


def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.


    Returns
    =======

    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    sim = {}

    for word, q_tf in query_word_counts.items():
        # skip words not in idf nor ii
        if word not in idf or word not in index:
            continue
        idf_val = idf[word]
        idf_square = idf_val * idf_val

        for posting in index[word]:
            doc_id, doc_tf = posting[0], posting[1]
            sim[doc_id] = sim.get(doc_id, 0) + q_tf * doc_tf * idf_square

    return sim


def index_search(
    query: str,
    index: dict,
    idf,
    doc_norms,
    score_func=accumulate_dot_scores,
    tokenizer=tokenize,
    roast_filter=None,
    max_price=None,
    min_score=None,
    beans=None
) -> List[Tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.

    tokenizer: a TreebankWordTokenizer

    roast_filter: list of strings, optional
        List of roast types to filter by. If provided, only beans with a matching
        roast_level will be included in the results.

    max_price: float, optional
        Maximum price per 100g in USD. Only beans with price <= max_price will be included.

    min_score: float, optional
        Minimum bean score. Only beans with score >= min_score will be included.

    beans: list of dictionaries, optional
        Original bean data needed for filtering. Required if roast_filter, max_price, or min_score is provided.

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.
    """
    # preprocess the query
    query = query.lower()
    tokens = tokenize(query)
    qcount = {}
    for token in tokens:
        qcount[token] = qcount.get(token, 0) + 1

    # compute the norm of the query
    qweight_sq = 0.0
    for word, count in qcount.items():
        if word in idf:
            qweight_sq += (idf[word] * count) ** 2
    query_norm = math.sqrt(qweight_sq) if qweight_sq > 0 else 0.0

    # get numerator dot prod dict for query
    dot_scores = score_func(qcount, index, idf)

    cosine_dict = {}
    for doc_id, dot_val in dot_scores.items():
        # Skip if the bean doesn't meet the filter criteria
        if beans and doc_id < len(beans):
            # Apply roast filter if specified
            if roast_filter and beans[doc_id]['roast'] not in roast_filter:
                continue

            # Apply price filter if specified
            if max_price is not None:
                bean_price = beans[doc_id].get('100g_USD')
                # Skip if price is missing or exceeds the maximum
                if bean_price is None or float(bean_price) > float(max_price):
                    continue

            # Apply score filter if specified
            if min_score is not None:
                bean_score = beans[doc_id].get('rating')
                # Skip if score is missing or below the minimum
                if bean_score is None or float(bean_score) < float(min_score):
                    continue

        cosine_dict[doc_id] = dot_val / (query_norm * doc_norms[doc_id]
                                         ) if query_norm > 0 and doc_norms[doc_id] > 0 else 0

    results = [(score, doc_id) for doc_id, score in cosine_dict.items()]
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:10]  # Limit to top 10 results after filtering


def filtered_search(query, inv_idx, idf, doc_norms, beans, roast_types=None, max_price=None, min_score=None):
    """Wrapper function to search with filtering

    Arguments:
    ==========
    query: string
        The query text to search for

    inv_idx: dict
        The inverted index

    idf: dict
        The precomputed IDF values

    doc_norms: list
        The precomputed document norms

    beans: list
        The original bean data

    roast_types: list, optional
        List of roast types to filter by

    max_price: float, optional
        Maximum price per 100g in USD

    min_score: float, optional
        Minimum bean score

    Returns:
    ========
    List of tuples (score, doc_id)
        Top 10 matching results after filtering
    """
    return index_search(query, inv_idx, idf, doc_norms,
                        roast_filter=roast_types, max_price=max_price,
                        min_score=min_score, beans=beans)


class SVDSearch:
    def __init__(self, beans, n_components=40):
        """Initialize the SVD search with coffee bean data

        Parameters:
        -----------
        beans: list
            List of coffee bean dictionaries
        n_components: int
            Number of latent dimensions for SVD
        """
        self.beans = beans
        self.n_components = n_components
        self.vectorizer = None
        self.td_matrix = None
        self.docs_compressed = None
        self.words_compressed = None
        self.singular_values = None
        self.docs_compressed_normed = None
        self.words_compressed_normed = None
        self.dimension_words = None

    def build_model(self):
        """Build the SVD model from bean descriptions"""
        combined_descriptions = [
            bean["desc_1"] + " " + bean["desc_2"] + " " + bean["desc_3"]
            for bean in self.beans
        ]

        # Use a custom analyzer that applies stemming
        self.vectorizer = TfidfVectorizer(
            # Use our custom tokenizer with stemming
            analyzer=lambda text: tokenize(text),
            stop_words='english',
            max_df=0.7,
            min_df=2
        )
        self.td_matrix = self.vectorizer.fit_transform(combined_descriptions)

        self.docs_compressed, self.singular_values, vt = svds(
            self.td_matrix, k=self.n_components)

        idx = np.argsort(-self.singular_values)
        self.singular_values = self.singular_values[idx]
        self.docs_compressed = self.docs_compressed[:, idx]
        vt = vt[idx, :]

        self.words_compressed = vt.transpose()

        self.docs_compressed_normed = normalize(self.docs_compressed)
        self.words_compressed_normed = normalize(self.words_compressed)

        self.dimension_words = self._get_top_words_per_dimension(top_n=3)

        return self
    

    def _get_top_words_per_dimension(self, top_n=3):
        feature_names = self.vectorizer.get_feature_names_out()
        dimension_words = []
        for dim in range(self.n_components):
            weights = self.words_compressed[:, dim]
            top_indices = np.argsort(np.abs(weights))[-top_n:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            dimension_words.append(top_words)
        return dimension_words
    

    def search(self, query, roast_types=None, max_price=None, min_score=None, k=10):
        """Search for beans similar to query in latent space

        Parameters:
        -----------
        query: str
            Query text to search for
        roast_types: list, optional
            List of roast types to filter by
        max_price: float, optional
            Maximum price per 100g in USD
        min_score: float, optional
            Minimum bean score
        k: int
            Number of results to return

        Returns:
        --------
        List of tuples (score, doc_id)
            Top k results sorted by similarity
        """
        # Use our stemming tokenizer for query processing
        query_tfidf = self.vectorizer.transform([query]).toarray()

        query_vec = np.dot(query_tfidf, self.words_compressed)

        query_vec = normalize(query_vec).squeeze()

        similarities = self.docs_compressed_normed.dot(query_vec)

        results = []
        for doc_id, sim in enumerate(similarities):
            if roast_types and self.beans[doc_id]['roast'] not in roast_types:
                continue

            if max_price is not None:
                bean_price = self.beans[doc_id].get('100g_USD')
                if bean_price is None or float(bean_price) > float(max_price):
                    continue

            if min_score is not None:
                bean_score = self.beans[doc_id].get('rating')
                if bean_score is None or float(bean_score) < float(min_score):
                    continue

            doc_vec = self.docs_compressed_normed[doc_id]
            latent_contributions = doc_vec * query_vec  
            latent_contributions = latent_contributions.tolist()  

            results.append((float(sim), doc_id, latent_contributions))

        # sort by similarity (descending)
        results.sort(reverse=True)

        return results[:k]
