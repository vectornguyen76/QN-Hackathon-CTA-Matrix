from transformers import AutoModel, AutoConfig
from preprocessing import Preprocess
import torch.nn as nn
import torch

from utils import pred_to_label

class ModelInference(nn.Module):
    def __init__(self, tokenizer, rdrsegmenter, model_path, checkpoint="vinai/phobert-base", device="cpu"):
        super(ModelInference, self).__init__()
        self.preprocess = Preprocess(tokenizer, rdrsegmenter)
        self.model = CustomModelSoftmax(checkpoint)
        self.device = device
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
        self.model.to(device)
    
    def predict(self, sample):
        self.model.eval()
        with torch.no_grad():
            # Clean input, segment and tokenize
            sample = self.preprocess.tokenize(sample)
            inputs = {"input_ids": sample["input_ids"].to(self.device),
                        "attention_mask": sample["attention_mask"].to(self.device)}

            # Predict
            outputs_classifier, outputs_regressor = self.model(**inputs)

            # Convert to numpy array
            outputs_classifier = outputs_classifier.cpu().numpy()
            outputs_regressor = outputs_regressor.cpu().numpy()

            # Get argmax each aspects
            outputs_regressor = outputs_regressor.argmax(axis=-1) + 1

            # Convert output to label
            outputs = pred_to_label(outputs_classifier, outputs_regressor)
        return outputs

class CustomModelSoftmax(nn.Module):
    def __init__(self, checkpoint):
        super(CustomModelSoftmax, self).__init__()
        self.model = model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768*4, 6)
        self.regressor = nn.Linear(768*4, 30)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)

        outputs = self.dropout(outputs)

        outputs_classifier = self.classifier(outputs)
        outputs_regressor = self.regressor(outputs)

        outputs_classifier = nn.Sigmoid()(outputs_classifier)
        outputs_regressor = outputs_regressor.reshape(-1, 6, 5)

        return outputs_classifier, outputs_regressor
