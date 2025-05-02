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

def parse_boolean_query(query: str) -> List[str]:
    """
    Parse a boolean query into tokens, handling AND, OR, NOT, and quoted phrases.
    Returns a list of tokens (terms and operators).
    """
    # Remove extra whitespace and normalize
    query = re.sub(r'\s+', ' ', query.strip())
    # Match quoted phrases, operators, parentheses, or words
    pattern = r'"(?:[^"]|"[^"]*")+"|\bAND\b|\bOR\b|\bNOT\b|\(|\)|[\w]+'
    tokens = re.findall(pattern, query, re.IGNORECASE)
    # Convert operators to uppercase and clean quotes from phrases
    cleaned_tokens = []
    for token in tokens:
        if token.upper() in ('AND', 'OR', 'NOT'):
            cleaned_tokens.append(token.upper())
        elif token.startswith('"') and token.endswith('"'):
            cleaned_tokens.append(token[1:-1])  # Remove quotes
        else:
            cleaned_tokens.append(token)
    return cleaned_tokens

def evaluate_boolean_query(tokens: List[str], inv_idx: dict, n_docs: int) -> set:
    """
    Evaluate a boolean query using a stack-based approach.
    Returns a set of document IDs matching the query.
    """
    def get_term_docs(term: str) -> set:
        """Get document IDs for a term or phrase."""
        terms = tokenize(term)
        if not terms:
            return set()
        # For phrases, take intersection of document IDs for all terms
        doc_sets = []
        for t in terms:
            if t in inv_idx:
                doc_sets.append(set(doc_id for doc_id, _ in inv_idx[t]))
        return set.intersection(*doc_sets) if doc_sets else set()

    stack = []
    operators = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                op = operators.pop()
                if op == 'NOT':
                    if stack:
                        operand = stack.pop()
                        stack.append(set(range(n_docs)) - operand)
                elif op in ('AND', 'OR'):
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(a & b if op == 'AND' else a | b)
            if operators and operators[-1] == '(':
                operators.pop()  # Remove '('
        elif token == 'NOT':
            operators.append(token)
        elif token in ('AND', 'OR'):
            # Handle precedence: NOT > AND > OR
            while operators and operators[-1] in ('NOT', 'AND') and (token != 'AND' or operators[-1] != 'NOT'):
                op = operators.pop()
                if op == 'NOT':
                    if stack:
                        operand = stack.pop()
                        stack.append(set(range(n_docs)) - operand)
                elif op == 'AND':
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(a & b)
            operators.append(token)
        else:
            # Token is a term or phrase
            stack.append(get_term_docs(token))
        i += 1

    # Process remaining operators
    while operators:
        op = operators.pop()
        if op == 'NOT':
            if stack:
                operand = stack.pop()
                stack.append(set(range(n_docs)) - operand)
        elif op in ('AND', 'OR'):
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a & b if op == 'AND' else a | b)

    return stack[0] if stack else set()

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
        The query we are looking for, may contain boolean operators AND, OR, NOT.

    index: an inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity.

    tokenizer: a TreebankWordTokenizer

    roast_filter: list of strings, optional
        List of roast types to filter by.

    max_price: float, optional
        Maximum price per 100g in USD.

    min_score: float, optional
        Minimum bean score.

    beans: list of dictionaries, optional
        Original bean data needed for filtering.

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results with highest score first.
    """
    # Parse query for boolean operators
    tokens = parse_boolean_query(query)
    n_docs = len(beans) if beans else len(doc_norms)

    if not tokens or all(token in ('AND', 'OR', 'NOT', '(', ')') for token in tokens):
        # Empty or operator-only query: score all documents
        matching_docs = set(range(n_docs))
        query_terms = []
    else:
        # Evaluate boolean query to get matching document IDs
        matching_docs = evaluate_boolean_query(tokens, index, n_docs)
        query_terms = [token for token in tokens if token not in ('AND', 'OR', 'NOT', '(', ')')]

    # If no matching documents, return empty results
    if not matching_docs:
        return []

    # Compute scores for matching documents
    # Build query vector from all terms (for scoring)
    query_text = ' '.join(query_terms) if query_terms else query
    query = query_text.lower()
    tokens = tokenizer(query)
    qcount = {}
    for token in tokens:
        qcount[token] = qcount.get(token, 0) + 1

    # Compute query norm
    qweight_sq = 0.0
    for word, count in qcount.items():
        if word in idf:
            qweight_sq += (idf[word] * count) ** 2
    query_norm = math.sqrt(qweight_sq) if qweight_sq > 0 else 0.0

    # Get dot product scores
    dot_scores = score_func(qcount, index, idf)

    cosine_dict = {}
    for doc_id in matching_docs:
        # Apply filters
        if beans and doc_id < len(beans):
            if roast_filter and beans[doc_id]['roast'] not in roast_filter:
                continue
            if max_price is not None:
                bean_price = beans[doc_id].get('100g_USD')
                if bean_price is None or float(bean_price) > float(max_price):
                    continue
            if min_score is not None:
                bean_score = beans[doc_id].get('rating')
                if bean_score is None or float(bean_score) < float(min_score):
                    continue

        # Compute cosine similarity
        dot_val = dot_scores.get(doc_id, 0)
        cosine_dict[doc_id] = dot_val / (query_norm * doc_norms[doc_id]) if query_norm > 0 and doc_norms[doc_id] > 0 else 0

    results = [(score, doc_id) for doc_id, score in cosine_dict.items()]
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:10]

def filtered_search(query, inv_idx, idf, doc_norms, beans, roast_types=None, max_price=None, min_score=None):
    """Wrapper function to search with filtering

    Arguments:
    ==========
    query: string
        The query text to search for, may include AND, OR, NOT

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
            Query text to search for, may include AND, OR, NOT
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
        List of tuples (score, doc_id, latent_contributions)
            Top k results sorted by similarity
        """
        tokens = parse_boolean_query(query)
        n_docs = len(self.beans)

        if not tokens or all(token in ('AND', 'OR', 'NOT', '(', ')') for token in tokens):
            results = []
            query_vec = np.mean(self.docs_compressed_normed, axis=0) 
            for doc_id in range(n_docs):
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
                similarity = float(doc_vec.dot(query_vec))
                latent_contributions = (doc_vec * query_vec).tolist()
                results.append((similarity, doc_id, latent_contributions))
            results.sort(reverse=True)
            return results[:k]

        stack = []
        operators = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    op = operators.pop()
                    if op == 'NOT':
                        if stack:
                            docs = stack.pop()
                            stack.append(set(range(n_docs)) - docs)
                    elif op in ('AND', 'OR'):
                        if len(stack) >= 2:
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(a & b if op == 'AND' else a | b)
                if operators and operators[-1] == '(':
                    operators.pop()
            elif token == 'NOT':
                operators.append(token)
            elif token in ('AND', 'OR'):
                while operators and operators[-1] in ('NOT', 'AND') and (token != 'AND' or operators[-1] != 'NOT'):
                    op = operators.pop()
                    if op == 'NOT':
                        if stack:
                            docs = stack.pop()
                            stack.append(set(range(n_docs)) - docs)
                    elif op == 'AND':
                        if len(stack) >= 2:
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(a & b)
                operators.append(token)
            else:
                # Token is a term or phrase
                query_tfidf = self.vectorizer.transform([token]).toarray()
                query_vec = np.dot(query_tfidf, self.words_compressed)
                query_vec = normalize(query_vec).squeeze()
                doc_ids = set()
                for doc_id in range(n_docs):
                    doc_vec = self.docs_compressed_normed[doc_id]
                    similarity = float(doc_vec.dot(query_vec))
                    if similarity > 0.1:  # Threshold for match
                        doc_ids.add(doc_id)
                stack.append(doc_ids)
            i += 1

        # Process remaining operators
        while operators:
            op = operators.pop()
            if op == 'NOT':
                if stack:
                    docs = stack.pop()
                    stack.append(set(range(n_docs)) - docs)
            elif op in ('AND', 'OR'):
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a & b if op == 'AND' else a | b)

        matching_docs = stack[0] if stack else set()
        if not matching_docs:
            return []

        query_terms = [token for token in tokens if token not in ('AND', 'OR', 'NOT', '(', ')')]
        query_text = ' '.join(query_terms) if query_terms else query
        query_tfidf = self.vectorizer.transform([query_text]).toarray()
        query_vec = np.dot(query_tfidf, self.words_compressed)
        query_vec = normalize(query_vec).squeeze()

        results = []
        for doc_id in matching_docs:
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
            similarity = float(doc_vec.dot(query_vec))
            latent_contributions = (doc_vec * query_vec).tolist()
            results.append((similarity, doc_id, latent_contributions))

        results.sort(reverse=True)
        return results[:k]