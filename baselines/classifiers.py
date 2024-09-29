import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim1=1024, hidden_dim2=512, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)   # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(hidden_dim2, num_classes)      # 第二个隐藏层到输出层          
        self.dropout = nn.Dropout(dropout)    # Dropout 防止过拟合

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.6)        # 第一个全连接层 + ReLU
        x = self.dropout(x)               # Dropout
        x = nn.functional.leaky_relu(self.fc2(x), negative_slope=0.6)        # 第二个全连接层 + ReLU
        x = self.dropout(x)               # Dropout
        return self.fc3(x)    


class BertBasedClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, feature_mode='text-only'):
        super(BertBasedClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        if feature_mode == 'text-only':
            self.fc1 = nn.Linear(self.bert.config.hidden_size * 2, 128)
        elif feature_mode == 'SFLP':
            self.fc1 = nn.Linear(self.bert.config.hidden_size * 2 + 60, 128)
        elif feature_mode == 'SFLP-max-mean' or feature_mode == 'SFLP-max-std' or feature_mode == 'SFLP-mean-std':
            self.fc1 = nn.Linear(self.bert.config.hidden_size * 2 + 40, 128)
        else:
            raise NotImplementedError(f"Feature mode {feature_mode} not implement yet!")
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.feature_mode = feature_mode
    
    def forward(self, input_ids, attention_mask, generated_input_ids, generated_attention_mask, top20_probs):
        # 文本输入特征
        input_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        input_pooled_output = input_output.pooler_output
        
        # 生成文本特征
        generated_output = self.bert(input_ids=generated_input_ids, attention_mask=generated_attention_mask)
        generated_pooled_output = generated_output.pooler_output
        
        if self.feature_mode == 'text-only':
            combined_features = torch.cat((input_pooled_output, generated_pooled_output), dim=1)
        elif self.feature_mode == 'SFLP':
            max_probs = top20_probs.max(dim=1).values
            top20_probs_stats = torch.cat((top20_probs.mean(dim=1), max_probs, top20_probs.std(dim=1)), dim=1)
            
            # 特征组合
            combined_features = torch.cat((input_pooled_output, generated_pooled_output, top20_probs_stats), dim=1)
        elif self.feature_mode == 'SFLP-max-mean':
            mean_probs = top20_probs.mean(dim=1)
            max_probs = top20_probs.max(dim=1).values
            top20_probs_stats = torch.cat((mean_probs, max_probs), dim=1)
            combined_features = torch.cat((input_pooled_output, generated_pooled_output, top20_probs_stats), dim=1)
        elif self.feature_mode == 'SFLP-max-std':
            max_probs = top20_probs.max(dim=1).values
            top20_probs_stats = torch.cat((max_probs, top20_probs.std(dim=1)), dim=1)
            combined_features = torch.cat((input_pooled_output, generated_pooled_output, top20_probs_stats), dim=1)
        elif self.feature_mode == 'SFLP-mean-std':
            top20_probs_stats = torch.cat((top20_probs.mean(dim=1), top20_probs.std(dim=1)), dim=1)
            combined_features = torch.cat((input_pooled_output, generated_pooled_output, top20_probs_stats), dim=1)
        else:
            raise NotImplementedError(f"Feature mode {self.feature_mode} not implement yet!")
        # 分类
        x = self.relu(self.fc1(combined_features))
        logits = self.fc2(x)
        return logits
    

class Bert_RNNBasedClassifier(nn.Module):
    def __init__(self, bert_model_name, rnn_hidden_dim, num_classes):
        super(Bert_RNNBasedClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.rnn = nn.LSTM(input_size=20, hidden_size=rnn_hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(self.bert.config.hidden_size * 2 + rnn_hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, generated_input_ids, generated_attention_mask, top20_probs):
        # BERT 部分
        input_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        input_pooled_output = input_output.pooler_output
        
        generated_output = self.bert(input_ids=generated_input_ids, attention_mask=generated_attention_mask)
        generated_pooled_output = generated_output.pooler_output
        
        # RNN 部分
        rnn_output, (hn, cn) = self.rnn(top20_probs)
        rnn_last_hidden = hn[-1]  # 获取最后一层的隐藏状态
        
        # 拼接特征
        combined_features = torch.cat((input_pooled_output, generated_pooled_output, rnn_last_hidden), dim=1)
        
        # 全连接层
        x = self.relu(self.fc1(combined_features))
        logits = self.fc2(x)
        return logits


