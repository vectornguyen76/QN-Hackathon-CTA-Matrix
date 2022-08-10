from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from model import ModelInference

rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = ModelInference(tokenizer, rdrsegmenter, 'model_softmax.pt')

app = Flask(__name__)

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/review-solver/solve")
def solve():
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]

    review_sentence = request.args.get('review_sentence')
    # review_sentence = "Bánh rất nhiều tôm to, tôm giòn nằm chễm chệ trên vỏ bánh mềm thơm ngon. Món ăn thuộc loại rolling in the deep, nghĩa là cuốn với rau, dưa chuột, giá, vỏ bánh mềm. Ngoài ra, đặc biệt không thể thiếu của món ăn là nước chấm chua cay rất Bình Định, vừa miệng đến khó tả. Đặc biệt, quán có sữa ngô tuyệt đỉnh, kết hợp Combo với bánh xèo cuốn này tạo thành một cặp trời sinh. Ai không thích tôm nhảy, có thể đổi sang bò hoặc mực cũng ngon không kém."
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