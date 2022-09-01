from transformers import AutoModel, AutoConfig, AutoTokenizer
from vncorenlp import VnCoreNLP
import torch.nn as nn
import numpy as np
import torch

from preprocessing import Preprocess
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

def score_to_tensor(score):
	tensor = np.zeros((score.shape[0], 6, 6))
	mask_up = np.ceil(score).reshape(-1).astype(np.int16)
	mask_down = np.floor(score).reshape(-1).astype(np.int16)
	xv, yv = np.meshgrid(np.arange(score.shape[0]), np.arange(6))
	y = yv.T.reshape(-1).astype(np.int16)
	x = xv.T.reshape(-1).astype(np.int16)
	score_up = (score - np.floor(score)).reshape(-1)
	score_down = (1 - score_up).reshape(-1)
	tensor[x, y, mask_up] = score_up
	tensor[x, y, mask_down] = score_down
	tensor[:,:,1] = tensor[:,:,0] + tensor[:,:,1]
	return tensor[:,:,1:]


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
		self.model = model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
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




class ScaledDotProductAttention(nn.Module):
	def __init__(self, d_k):
		super(ScaledDotProductAttention, self).__init__()
		self.d_k = d_k

	def forward(self, Q, K, V, attn_mask):
		# print(Q.shape, K.shape, V.shape, attn_mask.shape)
		scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
		scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
		attn = nn.Softmax(dim=-1)(scores)
		context = torch.matmul(attn, V)
		return context, attn

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, d_k,d_v, n_heads):
		super(MultiHeadAttention, self).__init__()
		self.d_model = d_model
		self.d_k = d_k
		self.d_v = d_v
		self.n_heads = n_heads
		
		self.W_Q = nn.Linear(d_model, d_k * n_heads)
		self.W_K = nn.Linear(d_model, d_k * n_heads)
		self.W_V = nn.Linear(d_model, d_v * n_heads)
		self.linear = nn.Linear(n_heads * d_v, d_model)
		self.layer_norm = nn.LayerNorm(d_model)

	def forward(self, Q, K, V, attn_mask):
		# q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
		residual, batch_size = Q, Q.size(0)
		# (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
		q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
		k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
		v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
		
		attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1)
		attn_mask = attn_mask.unsqueeze(-2).repeat(1, 1, Q.shape[1], 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
		# print(attn_mask.shape)
		# context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
		context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
		context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
		output = self.linear(context)
		return self.layer_norm(output + residual) # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
	def __init__(self, d_model, d_ff):
		super(PoswiseFeedForwardNet, self).__init__()
		self.d_model = d_model
		self.d_ff = d_ff
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		self.layer_norm = nn.LayerNorm(d_model)

	def forward(self, inputs):
		residual = inputs # inputs : [batch_size, len_q, d_model]
		output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
		output = self.conv2(output).transpose(1, 2)
		return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
	def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
		super(EncoderLayer, self).__init__()
		self.d_model = d_model
		self.d_k = d_k
		self.d_v = d_v
		self.n_heads = n_heads
		self.d_ff = d_ff
		self.enc_self_attn = MultiHeadAttention(d_model, d_k,d_v, n_heads)
		self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

	def forward(self, enc_inputs, enc_self_attn_mask):
		enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
		enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
		return enc_outputs

class CustomModelMultiHead(nn.Module):
	def __init__(self, checkpoint):
		super(CustomModelMultiHead, self).__init__()
		self.model = model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
		# for parameter in self.model.parameters():
		#   parameter.require_grad = False
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(768, 6)
		self.regressor = nn.Linear(768, 30)
		self.encoderlayer = EncoderLayer(768, 64, 64, 12, 3096)
	def forward(self, input_ids=None, attention_mask=None):
		outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
		outputs = self.dropout(outputs[0])
		outputs = self.encoderlayer(outputs, attention_mask.type(torch.bool)) # N, S, D
		outputs = outputs[:, 0, :]
		outputs_classifier = self.classifier(outputs)
		outputs_regressor = self.regressor(outputs)
		outputs_regressor = outputs_regressor.reshape(-1, 6, 5)
		outputs_classifier = nn.Sigmoid()(outputs_classifier)
		return outputs_classifier, outputs_regressor

class CustomModelMultiHeadRegressor(nn.Module):
	def __init__(self, checkpoint):
		super(CustomModelMultiHeadRegressor, self).__init__()
		self.model = model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
		# for parameter in self.model.parameters():
		#   parameter.require_grad = False
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(768, 6)
		self.regressor = nn.Linear(768, 6)
		self.encoderlayer = EncoderLayer(768, 64, 64, 12, 3096)
	def forward(self, input_ids=None, attention_mask=None):
		outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
		outputs = self.dropout(outputs[0])
		outputs = self.encoderlayer(outputs, attention_mask.type(torch.bool)) # N, S, D
		outputs = outputs[:, 0, :]
		outputs_classifier = self.classifier(outputs)
		outputs_regressor = self.regressor(outputs)
		outputs_classifier = nn.Sigmoid()(outputs_classifier)
		outputs_regressor = nn.Sigmoid()(outputs_regressor)*5
		return outputs_classifier, outputs_regressor



class ModelEnsemble(nn.Module):
	def __init__(self, tokenizer, rdrsegmenter, model_path1, model_path2, model_path3, model_path4,checkpoint="vinai/phobert-base", device="cpu"):
		super(ModelEnsemble, self).__init__()
		self.preprocess = Preprocess(tokenizer, rdrsegmenter)
		self.model1 = CustomModelSoftmax(checkpoint)
		self.model2 = CustomModel(checkpoint)
		self.model3 = CustomModelMultiHead(checkpoint)
		self.model4 = CustomModelMultiHeadRegressor(checkpoint)
		self.device = device
		self.model1.load_state_dict(torch.load(model_path1,map_location=torch.device(device)))
		self.model1.to(device)
		self.model2.load_state_dict(torch.load(model_path2,map_location=torch.device(device)))
		self.model2.to(device)
		self.model3.load_state_dict(torch.load(model_path3,map_location=torch.device(device)))
		self.model3.to(device)
		self.model4.load_state_dict(torch.load(model_path4,map_location=torch.device(device)))
		self.model4.to(device)

	def predict(self, sample):
		self.model1.eval()
		self.model2.eval()
		self.model3.eval()
		with torch.no_grad():
			sample = self.preprocess.tokenize(sample)
			inputs = {"input_ids": sample["input_ids"].to(self.device),
						"attention_mask": sample["attention_mask"].to(self.device)}
			outputs_classifier1, outputs_regressor1 = self.model1(**inputs)
			outputs_classifier2, outputs_regressor2 = self.model2(**inputs)
			outputs_classifier3, outputs_regressor3 = self.model3(**inputs)
			outputs_classifier4, outputs_regressor4 = self.model4(**inputs)


			outputs_classifier1 = outputs_classifier1.cpu().numpy()
			outputs_regressor1 = outputs_regressor1.cpu().numpy()
			outputs_classifier2 = outputs_classifier2.cpu().numpy()
			outputs_regressor2 = outputs_regressor2.cpu().numpy()
			outputs_classifier3 = outputs_classifier3.cpu().numpy()
			outputs_regressor3 = outputs_regressor3.cpu().numpy()
			outputs_classifier4 = outputs_classifier4.cpu().numpy()
			outputs_regressor4 = outputs_regressor4.cpu().numpy()

			outputs_regressor2 = score_to_tensor(outputs_regressor2)
			outputs_regressor4 = score_to_tensor(outputs_regressor4)

			outputs_regressor = (nn.Softmax(dim=-1)(torch.tensor(outputs_regressor1)).numpy() + outputs_regressor2 + nn.Softmax(dim=-1)(torch.tensor(outputs_regressor3)).numpy() + outputs_regressor4)/4
			outputs_classifier = (outputs_classifier1 + outputs_classifier2 + outputs_classifier3 + outputs_classifier4)/4
			outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
			outputs = pred_to_label(outputs_classifier, outputs_regressor)[0].astype(np.int32)
		return outputs.tolist()
  

def test():
    rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    model = ModelInference(tokenizer, rdrsegmenter, 'weights/model_softmax_v4.pt')
    # model = ModelEnsemble(tokenizer, rdrsegmenter, 'weights/model_softmax_v2_submit.pt', 'weights/model_regress_v2_submit.pt')
    print(model.predict("Các món ăn của nhà hàng rất dỡ mà còn mắc nữa, thái độ nhân viên rất tệ"))

if __name__ == "__main__":
	test()