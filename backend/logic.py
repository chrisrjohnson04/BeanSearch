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


def tokenize(text: str) -> List[str]:
    """Returns a list of words that make up the text.

    Parameters
    ----------
    text : str
        The input string to be tokenized.

    Returns
    -------
    List[str]
        A list of strings representing the words in the text.
    """
    return re.findall(r'\b[a-z]+\b', text.lower())


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
    # TODO-7.1
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

    beans: list of dictionaries, optional
        Original bean data needed for filtering. Required if roast_filter or max_price is provided.

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

        cosine_dict[doc_id] = dot_val / (query_norm * doc_norms[doc_id]
                                         ) if query_norm > 0 and doc_norms[doc_id] > 0 else 0

    results = [(score, doc_id) for doc_id, score in cosine_dict.items()]
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:10]  # Limit to top 10 results after filtering


def filtered_search(query, inv_idx, idf, doc_norms, beans, roast_types=None, max_price=None):
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

    Returns:
    ========
    List of tuples (score, doc_id)
        Top 10 matching results after filtering
    """
    return index_search(query, inv_idx, idf, doc_norms,
                        roast_filter=roast_types, max_price=max_price, beans=beans)


def main():
    # Sample queries
    queries = [u"A robust floral light roast bean with chocolate tones",
               u"Organic african bean from ethiopia"]

    # Load the transcript
    with open(os.path.join("coffee.json"), "r") as f:
        beans = json.load(f)

    # Precompute important values
    bean_tokens = tokenize_beans(beans)

    inv_idx = build_inverted_index(bean_tokens)

    idf = compute_idf(inv_idx, len(bean_tokens))

    inv_idx = {key: val for key, val in inv_idx.items()
               if key in idf}

    doc_norms = compute_doc_norms(inv_idx, idf, len(bean_tokens))

    # Compute results
    for query in queries:
        print("#" * len(query))
        print(query)
        print("#" * len(query))

        for score, bean_id in index_search(query, inv_idx, idf, doc_norms)[:10]:
            print("[{:.2f}] {}: {}\n\t({})".format(
                score,
                beans[bean_id]['name'],
                beans[bean_id]['origin_1'],
                beans[bean_id]['desc_1'] + " " + beans[bean_id]['desc_2'] + " " + beans[bean_id]['desc_3']))
        print()


if __name__ == '__main__':
    main()
