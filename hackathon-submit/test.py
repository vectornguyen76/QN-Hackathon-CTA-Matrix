import torch
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

text = "Chúng tôi làm việc ở Quy Nhơn"
output = rdrsegmenter.tokenize(text)
print(output)

# phobert = AutoModel.from_pretrained("vinai/phobert-large")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentence = 'Chúng_tôi là những nghiên_cứu_viên .'  

input_ids = torch.tensor([tokenizer.encode(sentence)])

print(input_ids)
# with torch.no_grad():
#     features = phobert(input_ids)  # Models outputs are now tuples