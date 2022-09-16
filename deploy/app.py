from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
from model import ModelInference

app = Flask(__name__)

# Segmenter input
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True)

# Get model
model = ModelInference(tokenizer, rdrsegmenter, 'weights/model.pt')

# API for build UI
@app.route('/', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
        
        # Get review from input
        review_sentence = request.form['input_review']

        # Predict
        predict_results = model.predict(review_sentence)

        results = dict()
        for i, aspect in enumerate(RATING_ASPECTS):
            results.update({aspect : str(predict_results[i]) + " ⭐"})

        return render_template("index.html", predict = results)
    
    else:
        return render_template("index.html")

# API for submit
@app.route("/review-solver/solve")
def solve():
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]

    # Get reviews
    review_sentence = request.args.get('review_sentence')
    
    # Model predict
    predict_results = model.predict(review_sentence)

    output = {
        "review": review_sentence,
        "results": {}
    }
    
    # Return json
    for count, r in enumerate(RATING_ASPECTS):
        output["results"][r] = int(predict_results[count])

    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)