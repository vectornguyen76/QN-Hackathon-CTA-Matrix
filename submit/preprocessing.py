import torch
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP


def segment(example, rdrsegmenter):
  return {"Segment": " ".join([" ".join(sen) for sen in rdrsegmenter.tokenize(example["Review"])])}

def tokenize(example, tokenizer):
  return tokenizer(example["Segment"], truncation=True, padding="max_length")



class Preprocess():
    def __init__(self, tokenizer, rdrsegmenter):
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
    def tokenize(self, sentence):
        segment = " ".join([" ".join(sen) for sen in self.rdrsegmenter.tokenize(sentence)])
        # print(segment)
        inputs = self.tokenizer(segment, truncation=True, return_tensors='pt')
        return inputs

if __name__ == "__main__":
    rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    preprocess = Preprocess(tokenizer, rdrsegmenter)
    sentences = "Chúng tôi là những nhà nghiên cứu"
    print(preprocess.tokenize(sentences))


    