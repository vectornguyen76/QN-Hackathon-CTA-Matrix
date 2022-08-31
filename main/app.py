from flask import Flask, request, jsonify, render_template
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from model import ModelEnsemble, ModelInference

app = Flask(__name__)

# Segmenter input
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True)
# model = ModelEnsemble(tokenizer, rdrsegmenter, 'weights/model_softmax_add_nha_hang_submit.pt', 'weights/model_regress_add_nha_hang_submit.pt', 'weights/model_attention_softmax_add_nha_hang_submit.pt', 'weights/model_attention_regress_add_nha_hang_submit.pt')
model = ModelInference(tokenizer, rdrsegmenter, 'weights/model_softmax_v4.pt')

@app.route('/', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':

        RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
        review_sentence = request.form['input_review']

        predict_results = model.predict(review_sentence)

        output = {
            "review": review_sentence,
            "results": {}
        }
        for count, r in enumerate(RATING_ASPECTS):
            output["results"][r] = int(predict_results[count])
        
        results = dict()
        for i, aspect in enumerate(RATING_ASPECTS):
            results.update({aspect : predict_results[i]})

        return render_template("index.html", predict = results)
    
    else:
        return render_template("index.html")


@app.route("/review-solver/solve")
def solve():
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]

    review_sentence = request.args.get('review_sentence')
    print("Review", review_sentence)
    predict_results = model.predict(review_sentence)

    output = {
        "review": review_sentence,
        "results": {}
      }
    for count, r in enumerate(RATING_ASPECTS):
        output["results"][r] = int(predict_results[count])

    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)