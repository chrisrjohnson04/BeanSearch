import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import logic
import nltk

# download NLTK data (only needed once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'coffee.json')

# Upload bean json
with open(os.path.join("coffee.json"), "r", encoding='utf-8') as f:
    beans = json.load(f)

app = Flask(__name__)
CORS(app)

# Precompute important values for search
bean_tokens = logic.tokenize_beans(beans)
inv_idx = logic.build_inverted_index(bean_tokens)
idf = logic.compute_idf(inv_idx, len(bean_tokens))
inv_idx = {key: val for key, val in inv_idx.items() if key in idf}
doc_norms = logic.compute_doc_norms(inv_idx, idf, len(bean_tokens))


# Initialize and build SVD model
print("Building SVD model...")
svd_search = logic.SVDSearch(beans, n_components=40)
svd_search.build_model()
print("SVD model built successfully!")


def json_search(query, roast_types=None, max_price=None, min_score=None, use_svd=False):
    """Search function that supports roast, price, and score filtering,
    with option to use SVD-based semantic search"""

    if use_svd:
        print(f"Using SVD search with query: '{query}'")
        search_results = svd_search.search(
            query, roast_types, max_price, min_score)
    else:
        print(f"Using traditional search with query: '{query}'")
        search_results = logic.filtered_search(
            query, inv_idx, idf, doc_norms, beans, roast_types, max_price, min_score)

    res = []
    for result in search_results:
        if use_svd:
            score, bean_id, latent_contributions = result
        else:
            score, bean_id = result
            latent_contributions = None
        obj = beans[bean_id]
        bean_copy = obj.copy()
        bean_copy["desc"] = obj['desc_1'] + " " + \
            obj['desc_2'] + " " + obj['desc_3']
        # Include the score for debugging purposes
        bean_copy["match_score"] = round(score, 4)
        if latent_contributions is not None:
            bean_copy["latent_contributions"] = [round(contrib, 4) for contrib in latent_contributions]
        res.append(bean_copy)

    response = {
        "results": res,
        "dimension_words": svd_search.dimension_words if use_svd else None
    }

    return json.dumps(response)


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/beans")
def beans_search():
    text = request.args.get("bean_query", "")
    roast_types_param = request.args.get("roast_types", "")
    max_price = request.args.get("max_price")
    min_score = request.args.get("min_score")
    use_svd = request.args.get("use_svd", "false").lower() == "true"

    # Parse roast types from URL params
    roast_types = roast_types_param.split(',') if roast_types_param else []
    roast_types = [rt for rt in roast_types if rt]

    # Convert max_price to float if provided
    if max_price:
        try:
            max_price = float(max_price)
        except ValueError:
            max_price = None

    # Convert min_score to float if provided
    if min_score:
        try:
            min_score = float(min_score)
        except ValueError:
            min_score = None

    return json_search(text, roast_types if roast_types else None, max_price, min_score, use_svd)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
