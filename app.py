from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore")

model_name = 'Sakil/sentence_similarity_semantic_search'
app = Flask(__name__)
model = SentenceTransformer(model_name)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/text-similarity', methods=['POST'])
@app.route('/text-similarity', methods=['POST'])
def text_similarity():
    try:
        data = request.get_json()
        text1 = data['text1']
        text2 = data['text2']

        print(f"Received text1: {text1}")
        print(f"Received text2: {text2}")

        similarity_score = predict(text1, text2, model)
        response = {
            'similarity_score': float(similarity_score),  # Convert to float
            'message': 'Text similarity calculated successfully'
        }

        print(f"Calculated similarity score: {similarity_score}")

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        error_response = {
            'error': str(e),
            'message': 'Error in calculating text similarity'
        }
        return jsonify(error_response), 500

"""
def text_similarity():
    try:
        data = request.get_json()
        text1 = data['text1']
        text2 = data['text2']

        similarity_score = predict(text1, text2, model)
        response = {
            'similarity_score': similarity_score.item(),  # Convert to float
            'message': 'Text similarity calculated successfully'
        }

        return jsonify(response)

    except Exception as e:
        error_response = {
            'error': str(e),
            'message': 'Error in calculating text similarity'
        }
        return jsonify(error_response), 500
"""

def predict(sentence1, sentence2, model):
    emb1 = model.encode(sentence1)
    emb2 = model.encode(sentence2)
    return util.cos_sim(emb1, emb2)

if __name__ == '__main__':
    app.run(port=5000)
