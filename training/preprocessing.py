import numpy as np
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

def clean_text(text):
	return {"Review": normalize_annotatation(remove_number(remove_special_char(remove_punctuation(strip_emoji(text["Review"].lower())))))}

class Preprocess():
    def __init__(self, tokenizer, rdrsegmenter):
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
        self.feature = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']

    def segment(self, example):
        return {"Segment": " ".join([" ".join(sen) for sen in self.rdrsegmenter.tokenize(example["Review"])])}
 
    def tokenize(self, example):
        return self.tokenizer(example["Segment"], truncation=True)
    
    def label(self, example):
        return {'labels_regressor': np.array([example[i] for i in self.feature]),
            'labels_classifier': np.array([int(example[i] != 0) for i in self.feature])}
        
    def run(self, dataset):
        dataset = dataset.map(clean_text)
        dataset = dataset.map(self.segment)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset = dataset.map(self.label)
        dataset = dataset.remove_columns(['Unnamed: 0','Review', 'giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam', 'Segment'])
        dataset.set_format("torch")
        
        return dataset


    