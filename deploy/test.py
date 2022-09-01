import torch
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP

# Segmenter input
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentence = 'Chúng_tôi là những nghiên_cứu_viên .'  
input_ids = torch.tensor([tokenizer.encode(sentence)])

print(input_ids)