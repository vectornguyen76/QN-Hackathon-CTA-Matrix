import torch
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
import re

def strip_emoji(text):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    return RE_EMOJI.sub(r'', text)

def remove_special_char(text):
  special_character = re.compile("�+")
  return special_character.sub(r'', text)

def remove_punctuation(text):
  punctuation = re.compile(r"[!#$%&()*+;<=>?@[\]^_`{|}~]")
  return punctuation.sub(r"", text)

def remove_number(text):
  return re.sub(" \d+", " ", text)

def normalize_annotatation(text):
  khach_san = "\bkhach san ?|\bksan ?|\bks ?"
  return re.sub("\bnv ?", "nhân viên",re.sub(khach_san, "khách sạn", text))

def segment(example, rdrsegmenter):
  return {"Segment": " ".join([" ".join(sen) for sen in rdrsegmenter.tokenize(example["Review"])])}

def tokenize(example, tokenizer):
  return tokenizer(example["Segment"], truncation=True, padding="max_length")

def clean_text(text):
  return normalize_annotatation(remove_number(remove_special_char(remove_punctuation(strip_emoji(text.lower())))))

class Preprocess():
    def __init__(self, tokenizer, rdrsegmenter):
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
    def tokenize(self, sentence):
        sentence = clean_text(sentence)
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


    