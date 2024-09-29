from baselines import *
from easyeditor import KEIDDataset
from torch.utils.data import DataLoader, TensorDataset
import argparse
import pickle
import logging
import os
import torch
torch.manual_seed(42)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from transformers import AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

type_mapping = {
    "Non edited": 0,
    "Fact updating": 1,
    "Misinformation injection": 2,
    "Offensiveness injection": 3,
    "Behavioral misleading injection": 4,
    "Bias injection": 5
}
llm_mapping = {
    "Meta-Llama-3.1-8B-Instruct": "llama3.1-8b",
    "Llama-2-13b-chat-hf": "llama2-13b"
}

LOG = logging.getLogger(__name__)

def preprocess(tokens_log_probs, edited_llm_tokenizer):
    generated_texts, log_probs = [], []
    for items in tokens_log_probs:
        top20_probs = []
        generated_tokens = []
        for item in items:
            generated_tokens.append(item['token_ids'][0])
            top20_probs.append(item['logprobs'])
        log_probs.append(torch.stack(top20_probs))
        generated_texts.append(edited_llm_tokenizer.decode(generated_tokens))
    return generated_texts, torch.stack(log_probs)

# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            generated_input_ids = batch['generated_input_ids'].to(device)
            generated_attention_mask = batch['generated_attention_mask'].to(device)
            top20_probs = batch['top20_probs'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, generated_input_ids, generated_attention_mask, top20_probs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        LOG.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            generated_input_ids = batch['generated_input_ids'].to(device)
            generated_attention_mask = batch['generated_attention_mask'].to(device)
            top20_probs = batch['top20_probs'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, generated_input_ids, generated_attention_mask, top20_probs)
            _, preds = torch.max(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, precision, recall, f1, cm

# 绘制混淆矩阵
def plot_confusion_matrix(cm, class_names, save_path):
    num_classes = len(class_names)
    c_shape = cm.shape
    if c_shape != (num_classes, num_classes):
        padded_matrix = np.zeros((num_classes, num_classes), dtype=int)
        padded_matrix[:c_shape[0], :c_shape[1]] = cm
        cm = padded_matrix

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='BERT+LSTM', choices=['BERT', 'BERT+LSTM'], type=str)
    parser.add_argument('--edit_method', default='ft',choices=['ft','grace','unke'], type=str)
    parser.add_argument('--edited_llm', default='Meta-Llama-3.1-8B-Instruct', choices=llm_mapping.keys(), type=str)
    parser.add_argument('--feature_dir', default='./features', type=str)
    parser.add_argument('--more_tokens', default=6, type=int)
    parser.add_argument('--log_level', default='INFO', type=str)
    parser.add_argument('--rephrased', default=False, action="store_true")
    parser.add_argument('--test_feature', default=None,  choices=['ft','grace','unke', 'non-edit'])
    parser.add_argument('--pretrained_model_path', default="/science/llms/", type=str)
    parser.add_argument('--feature_mode', default=None, choices=['text-only','SFLP','SFLP-max-mean','SFLP-max-std','SFLP-mean-std'], type=str)


    args = parser.parse_args()

    log_level = logging.INFO
    if args.log_level == 'DEBUG':
        log_level = logging.DEBUG
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = log_level)
    if args.method == "BERT":
        if args.feature_mode is None:
            raise ValueError(f"Feature mode should be specificted.")

    test_set = KEIDDataset('./data/edit_intention_split/test.json', type_mapping)
    train_set = KEIDDataset('./data/edit_intention_split/train.json', type_mapping)
    train_queries, test_queries, train_labels, test_labels = [], [], [], []
    for query, rephrased_query, label in train_set:
        if args.rephrased:
            train_queries.append(rephrased_query)
        else:
            train_queries.append(query)
        train_labels.append(label)
    for query, rephrased_query, label in test_set:
        if args.rephrased:
            test_queries.append(rephrased_query)
        else:
            test_queries.append(query)
        test_labels.append(label)
        if args.test_feature is not None and args.test_feature=="non-edit":
            test_labels = list(0*np.array(test_labels))


    LOG.info("Preprocess features..")
    LOG.info("Load features..")
    if args.test_feature is not None:
        test_feature_dir = f"{args.feature_dir}/{args.test_feature}_{llm_mapping[args.edited_llm]}_testset_token_{args.more_tokens}{'_rephrased' if args.rephrased and args.test_feature!='non-edit' else ''}.pkl"
    else:
        test_feature_dir = f"{args.feature_dir}/{args.edit_method}_{llm_mapping[args.edited_llm]}_testset_token_{args.more_tokens}{'_rephrased' if args.rephrased else ''}.pkl"
    if os.path.isfile(test_feature_dir):
        with open(test_feature_dir, 'rb') as f:
            test_feature = pickle.load(f)
    else:
        raise ValueError(f"File {test_feature_dir} isn't exsit.")
    train_feature_dir = f"{args.feature_dir}/{args.edit_method}_{llm_mapping[args.edited_llm]}_trainset_token_{args.more_tokens}{'_rephrased' if args.rephrased else ''}.pkl"
    if os.path.isfile(train_feature_dir):
        with open(train_feature_dir, 'rb') as f:
            train_feature = pickle.load(f)
    else:
        raise ValueError(f"File {train_feature_dir} isn't exsit.")
    LOG.info("Load features done.")

    all_test_tokens_log_probs = test_feature['top_k_tokens_log_probs']
    all_train_tokens_log_probs = train_feature['top_k_tokens_log_probs']
    edited_llm_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path + args.edited_llm)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path + 'bert-base-uncased')

    generated_texts, top20_probs = preprocess(all_test_tokens_log_probs, edited_llm_tokenizer)
    feature_test = ClosedLLMFeatureDataset(query=test_queries, generated_texts=generated_texts, top20_probs=top20_probs, \
                                            labels=test_labels, tokenizer=tokenizer)
    generated_texts, top20_probs = preprocess(all_train_tokens_log_probs, edited_llm_tokenizer)
    feature_train = ClosedLLMFeatureDataset(query=train_queries, generated_texts=generated_texts, top20_probs=top20_probs, \
                                            labels=train_labels, tokenizer=tokenizer)
    
    train_dataloader = DataLoader(feature_train, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(feature_test, batch_size=8, shuffle=False)
    LOG.info("Preprocess done.")
    if args.method == "BERT":
        model = BertBasedClassifier(args.pretrained_model_path + 'bert-base-uncased', feature_mode=args.feature_mode, num_classes=6).to(device)
    else:
        model = Bert_RNNBasedClassifier(args.pretrained_model_path + 'bert-base-uncased', rnn_hidden_dim=256, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    LOG.info("Train start.")
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=6)

    LOG.info("Train done.")
    LOG.info("Start eval.")
    model.eval()
    accuracy, precision, recall, f1, cm = evaluate_model(model, test_dataloader)

    eval_res = f'Accuracy: {accuracy:.3f}\n' + f'Precision: {precision:.3f}\n' + f'Recall: {recall:.3f}\n' + f'F1 Score: {f1:.3f}\n' + f'Confusion Matrix:\n{cm}'
    with open(f"./results/{args.method}_{args.edit_method}{f'_{args.feature_mode}'if args.feature_mode is not None else ''}{f'_to_{args.test_feature}' if args.test_feature is not None else ''}_{args.edited_llm}{'_rephrased' if args.rephrased else ''}.txt", 'w') as file:  # 'w' 模式会覆盖文件内容，'a' 模式是追加内容
        file.write(eval_res)
    print(eval_res)


    # 2. 输出混淆矩阵
    plot_confusion_matrix(cm, ['NE', "FU", "MI", "OI", "BMI", "BI"], f"./figs_new/cm_{args.method}_{args.edit_method}{f'_{args.feature_mode}'if args.feature_mode is not None else ''}{f'_to_{args.test_feature}' if args.test_feature is not None else ''}_{args.edited_llm}{'_rephrased' if args.rephrased else ''}.pdf")
