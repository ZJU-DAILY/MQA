import json
import sys
from functools import wraps

import numpy as np
from flask import Flask, request, jsonify, Blueprint, send_file
import os

from pojo.dataset import get_modal_type
from pojo.embedding import Embedding
from pojo.index import get_index, set_index
from pojo.response_data import ResponseData
from pojo.search import get_search
from vector_weight_learning import fvecs_converter

blueprint = Blueprint('blueprint', __name__, url_prefix='/m1/4132394-0-default')


# a decorator to simplify response
def jsonify_response_data(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response_data = f(*args, **kwargs)
        if isinstance(response_data, ResponseData):
            response_dict = response_data.to_dict()
            return jsonify(response_dict)
        return response_data

    return decorated_function


@blueprint.route('/embedding', methods=['POST'])
@jsonify_response_data
def post_embedding():
    try:
        body = request.json
        learning = body['learning']
        data = {
            'id': 0,
            'modalities': body['modalities'],
            'deleted': False
        }

        weight = []
        for modality in data['modalities']:
            modality['weight'] = 1.0 / len(data['modalities'])
            weight.append(1.0 / len(data['modalities']))

        embedding = Embedding(data=data)
        embedding.create_embedding(data=data)
        if learning and len(data['modalities']) > 1:
            weight = embedding.learning()
            # return ResponseData(data=weight)
        return ResponseData(data={"weight": weight})
    except Exception as e:
        return ResponseData(message=str(e), data={})


@blueprint.route('/index', methods=['POST'])
@jsonify_response_data
def post_index():
    body = request.json
    algorithm = body['algorithm']
    neighbor = body['neighbor']
    candidate = body['candidate']
    index_weight = body['index_weight']

    # normalize indexes' weight
    total = sum(index_weight)
    index_weight = [item / total for item in index_weight]

    set_index(algorithm=algorithm, neighbor=neighbor, candidate=candidate, index_weight=index_weight)
    return ResponseData(data={})


@blueprint.route('/search', methods=['POST'])
@jsonify_response_data
def post_search():
    try:
        print(request.form)
        llm = request.form.get('llm')
        text = request.form.get('text')
        temperature = float(request.form.get('temperature'))
        retrieval_number = int(request.form.get('resultNumber'))
        retrieval_framework = request.form.get('retrievalFramework')
        use_knowledge = request.form.get('useKnowledge')
        selected_target = int(request.form.get('selectedTarget'))
        retrieval_weight = [float(item) for item in json.loads(request.form.get('retrievalWeight'))]

        search = None
        text_path = ""
        if use_knowledge == 'true' and llm != 'dall-e-3':
            for filename in os.listdir(search_path):
                if filename != 'result.txt':
                    filepath = os.path.join(search_path, filename)
                    os.remove(filepath)

            # save file to /search/xxx.tmp
            text_path = ''
            embedding = Embedding.get()
            for i, modality in enumerate(embedding.modalities):
                for j, modal in enumerate(modality.modals):
                    t = get_modal_type(modal)
                    with open(os.path.join(search_path, f'{modal}.tmp'), 'w') as file:
                        if t == 'text':
                            text_path = os.path.join(search_path, f'{modal}.tmp')
                            file.write(text)
                        elif t in request.files:
                            tmp = request.files[t]
                            tmp.save(os.path.join(upload_path, tmp.filename))
                            file.write(os.path.join(upload_path, tmp.filename))

            # fix weight
            index_method, index_path = get_index()
            if len(retrieval_weight) != 0:
                total = sum(retrieval_weight)
                retrieval_weight = [item / total for item in retrieval_weight]
            search = get_search(retrieval_framework=retrieval_framework,
                       selected_target=selected_target,
                       retrieval_number=retrieval_number,
                       retrieval_weight=retrieval_weight,
                       index_path=index_path,
                       index_method=index_method)

        # get LLM
        from pojo.llm import get_llm
        llm_model = get_llm(model=llm, temperature=temperature, history=history, search=search, text_path=text_path)
        data = llm_model.generate_answer(content=text)
        return ResponseData(data=data)

    except Exception as e:
        return ResponseData(message=str(e), data={})


@blueprint.route('/image', methods=['GET'])
@jsonify_response_data
def get_image():
    param = request.args
    meta = param.get('meta')
    id = int(param.get('id'))

    with open(os.path.join(root, 'dataset', 'meta', f'{meta}.txt'), 'r') as file:
        for line_no, line in enumerate(file):
            if line_no == id:
                return send_file(line.strip(), mimetype='image/jpeg')


app = Flask(__name__)
app.register_blueprint(blueprint)


# set CORS headers after request
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response


@app.errorhandler(KeyError)
@jsonify_response_data
def handle_key_error(e):
    return ResponseData(message=str(e), data={})


if __name__ == '__main__':
    root = os.getcwd()
    if sys.platform.startswith('win'):
        root = '\\\\?\\' + root

    dataset_path = os.path.join(root, 'dataset')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    base_path = os.path.join(dataset_path, 'base')
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    # index_path = os.path.join(dataset_path, 'index')
    # if not os.path.exists(index_path):
    #     os.mkdir(index_path)

    meta_path = os.path.join(dataset_path, 'meta')
    if not os.path.exists(meta_path):
        os.mkdir(meta_path)

    meta_path = os.path.join(dataset_path, 'query')
    if not os.path.exists(meta_path):
        os.mkdir(meta_path)

    search_path = os.path.join(dataset_path, 'search')
    if not os.path.exists(search_path):
        os.mkdir(search_path)

    upload_path = os.path.join(root, 'uploads')
    if not os.path.exists(upload_path):
        os.mkdir(upload_path)

    embedding_config = os.path.join(dataset_path, 'config.json')
    if not os.path.exists(embedding_config):
        with open(embedding_config, 'w'):
            pass

    delete_id_path = os.path.join(dataset_path, 'delete.ivecs')
    fvecs_converter.to_fvecs(delete_id_path, [[]])

    history = []

    app.run(host='127.0.0.1', port=4523, debug=True)
