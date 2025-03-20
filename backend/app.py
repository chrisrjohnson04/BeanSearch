import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import logic

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'coffee.json')

# Upload bean json
with open(os.path.join("coffee.json"), "r") as f:
    beans = json.load(f)

app = Flask(__name__)
CORS(app)


def json_search(query):
    bean_tokens = logic.tokenize_beans(beans)

    inv_idx = logic.build_inverted_index(bean_tokens)

    idf = logic.compute_idf(inv_idx, len(bean_tokens))

    inv_idx = {key: val for key, val in inv_idx.items()
               if key in idf}

    doc_norms = logic.compute_doc_norms(inv_idx, idf, len(bean_tokens))

    res = []

    for score, bean_id in logic.index_search(query, inv_idx, idf, doc_norms)[:10]:
        obj = beans[bean_id]
        obj["desc"] = beans[bean_id]['desc_1'] + " " + \
            beans[bean_id]['desc_2'] + " " + beans[bean_id]['desc_3']
        res.append(obj)

    return json.dumps(res)


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/beans")
def episodes_search():
    text = request.args.get("bean_query")
    return json_search(text)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
