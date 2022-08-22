from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from model import ModelInference

rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = ModelInference(tokenizer, rdrsegmenter, 'weights/model_softmax_v4.pt')

app = Flask(__name__)

@app.route("/")
def root():
    return {"message": "Hello World"}


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
        output["results"][r] = predict_results[count]

    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)