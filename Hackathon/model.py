import torch.nn as nn
import torch
from transformers import AutoModel, AutoConfig
from preprocessing import Preprocess
from utils import pred_to_label
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
import numpy as np


class CustomModelRegressor(nn.Module):
  def __init__(self, checkpoint, num_outputs):
    super(CustomModelRegressor, self).__init__()
    self.num_outputs = num_outputs
    self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    for parameter in self.model.parameters():
      parameter.require_grad = False
    self.dropout = nn.Dropout(0.1)
    self.output1 = nn.Linear(768*4, 6)
    # self.output2 = nn.Linear(96, 6)
  def forward(self, input_ids=None, attention_mask=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs = self.dropout(outputs_sequence[2])
    outputs = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
    outputs = self.dropout(outputs)
    # outputs = self.output1(outputs[:, 0, :])
    outputs = self.output1(outputs)
    # outputs = self.output2(outputs)
    outputs = nn.Sigmoid()(outputs)*5
    return outputs

class CustomModelClassifier(nn.Module):
  def __init__(self, checkpoint, num_outputs):
    super(CustomModelClassifier, self).__init__()
    self.num_outputs = num_outputs
    self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    for parameter in self.model.parameters():
      parameter.require_grad = False
    self.dropout = nn.Dropout(0.1)
    self.output1 = nn.Linear(768*4, 30)
    # self.output2 = nn.Linear(96, 6)
  def forward(self, input_ids=None, attention_mask=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs = self.dropout(outputs_sequence[2])
    outputs = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
    outputs = self.dropout(outputs)
    # outputs = self.output1(outputs[:, 0, :])
    outputs = self.output1(outputs)
    # outputs = self.output2(outputs)
    # outputs = nn.Sigmoid()(outputs)*5
    return outputs

class CustomModel(nn.Module):
  def __init__(self, checkpoint):
    super(CustomModel, self).__init__()
    # self.num_outputs = num_outputs
    self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    # for parameter in self.model.parameters():
    #   parameter.require_grad = False
    self.dropout = nn.Dropout(0.4)
    self.classifier = nn.Linear(768*4, 6)
    self.regressor = nn.Linear(768*4, 6)
    # self.output2 = nn.Linear(96, 6)
  def forward(self, input_ids=None, attention_mask=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs = self.dropout(outputs_sequence[2])
    outputs = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
    outputs = self.dropout(outputs)
    # outputs = self.output1(outputs[:, 0, :])
    outputs_classifier = self.classifier(outputs)
    outputs_regressor = self.regressor(outputs)
    # outputs = self.output2(outputs)
    outputs_classifier = nn.Sigmoid()(outputs_classifier)
    outputs_regressor = nn.Sigmoid()(outputs_regressor)*5
    return outputs_classifier, outputs_regressor

class CustomModelSoftmax(nn.Module):
  def __init__(self, checkpoint):
    super(CustomModelSoftmax, self).__init__()
    # self.num_outputs = num_outputs
    self.model = model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    # for parameter in self.model.parameters():
    #   parameter.require_grad = False
    self.dropout = nn.Dropout(0.1)
    self.classifier = nn.Linear(768*4, 6)
    self.regressor = nn.Linear(768*4, 30)
    # self.output2 = nn.Linear(96, 6)
  def forward(self, input_ids=None, attention_mask=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs = self.dropout(outputs_sequence[2])
    outputs = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
    outputs = self.dropout(outputs)
    # outputs = self.output1(outputs[:, 0, :])
    outputs_classifier = self.classifier(outputs)
    outputs_regressor = self.regressor(outputs)
    outputs_regressor = outputs_regressor.reshape(-1, 6, 5)
    # outputs = self.output2(outputs)
    outputs_classifier = nn.Sigmoid()(outputs_classifier)
    # outputs_regressor = nn.Sigmoid()(outputs_regressor)*5
    return outputs_classifier, outputs_regressor

class CustomModelMultiHead(nn.Module):
  def __init__(self, checkpoint):
    super(CustomModelMultiHead, self).__init__()
    # self.num_outputs = num_outputs
    self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    for parameter in self.model.parameters():
      parameter.require_grad = False
    self.dropout = nn.Dropout(0.1)
    self.reduce = nn.Linear(768, 16)
    self.flatten = nn.Flatten()
    self.classifier = nn.Linear(16*256, 6)
    self.regressor = nn.Linear(16*256, 30)
    self.multihead = nn.MultiheadAttention(768, 4, batch_first=True)
    self.ff = nn.Linear(768, 768)
    self.norm1 = nn.LayerNorm(768)
    self.norm2 = nn.LayerNorm(768)
    # self.output2 = nn.Linear(96, 6)
  def forward(self, input_ids=None, attention_mask=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs = self.dropout(outputs_sequence[2])
    # outputs = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
    outputs = self.dropout(outputs[0])
    inputs = outputs
    # print(outputs.shape)
    # print(attention_mask.shape)
    outputs = self.multihead(outputs, outputs, outputs, key_padding_mask=attention_mask.float())[0]
    outputs = inputs + outputs
    outputs = self.norm1(outputs)
    inputs = outputs
    outputs = nn.ReLU()(self.ff(outputs))
    outputs = self.norm2(inputs + outputs)
    outputs = self.reduce(outputs)
    outputs = self.flatten(outputs)
    # outputs = self.dropout(outputs)
    # outputs = self.output1(outputs[:, 0, :])
    outputs_classifier = self.classifier(outputs)
    outputs_regressor = self.regressor(outputs)
    outputs_regressor = outputs_regressor.reshape(-1, 6, 5)
    # outputs = self.output2(outputs)
    outputs_classifier = nn.Sigmoid()(outputs_classifier)
    # outputs_regressor = nn.Sigmoid()(outputs_regressor)*5
    return outputs_classifier, outputs_regressor

class ModelInference(nn.Module):
    def __init__(self, tokenizer, rdrsegmenter, model_path,checkpoint="vinai/phobert-base", device="cpu"):
        super(ModelInference, self).__init__()
        self.preprocess = Preprocess(tokenizer, rdrsegmenter)
        self.model = CustomModelSoftmax(checkpoint)
        self.device = device
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
        self.model.to(device)
    def predict(self, sample):
        self.model.eval()
        with torch.no_grad():
            sample = self.preprocess.tokenize(sample)
            inputs = {"input_ids": sample["input_ids"].to(self.device),
                        "attention_mask": sample["attention_mask"].to(self.device)}
            outputs_classifier, outputs_regressor = self.model(**inputs)
            outputs_classifier = outputs_classifier.cpu().numpy()
            outputs_regressor = outputs_regressor.cpu().numpy()
            outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
            outputs = pred_to_label(outputs_classifier, outputs_regressor)[0].astype(np.int32)
        return outputs.tolist()

if __name__ == "__main__":
    rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = ModelInference(tokenizer, rdrsegmenter, 'weights/model_softmax.pt')
    print(model.predict("Bánh rất nhiều tôm to, tôm giòn nằm chễm chệ trên vỏ bánh mềm thơm ngon. Món ăn thuộc loại rolling in the deep, nghĩa là cuốn với rau, dưa chuột, giá, vỏ bánh mềm. Ngoài ra, đặc biệt không thể thiếu của món ăn là nước chấm chua cay rất Bình Định, vừa miệng đến khó tả. Đặc biệt, quán có sữa ngô tuyệt đỉnh, kết hợp Combo với bánh xèo cuốn này tạo thành một cặp trời sinh. Ai không thích tôm nhảy, có thể đổi sang bò hoặc mực cũng ngon không kém."))

