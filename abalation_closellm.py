from baselines import *
from easyeditor import KEIDDataset
from torch.utils.data import DataLoader, TensorDataset
import argparse
import pickle
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
from transformers import AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import gc

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
    "Llama-2-13b-chat-hf": "llama2-13b",
    # "Meta-Llama-3.1-8B-Instruct": "llama3.1-8b",

}

LOG = logging.getLogger(__name__)


def preprocess(tokens_log_probs, edited_llm_tokenizer, token_num):
    generated_texts, log_probs = [], []
    for items in tokens_log_probs:
        top20_probs = []
        generated_tokens = []
        for item in items:
            generated_tokens.append(item['token_ids'][0])
            top20_probs.append(item['logprobs'])
        log_probs.append(torch.stack(top20_probs[:token_num]))
        generated_texts.append(edited_llm_tokenizer.decode(generated_tokens[:token_num]))
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

def plot_confusion_matrix(cm, class_names, save_path):
    num_classes = len(class_names)
    c_shape = cm.shape
    if c_shape != (num_classes, num_classes):
        padded_matrix = np.zeros((num_classes, num_classes), dtype=int)
        padded_matrix[:c_shape[0], :c_shape[1]] = cm
        cm = padded_matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)


args = argparse.Namespace(
    method='BERT+LSTM',
    feature_dir='./features',
    log_level='INFO',
    rephrased=False,
    pretrained_model_path='/science/llms/',
)

log_level = logging.INFO
if args.log_level == 'DEBUG':
    log_level = logging.DEBUG
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S',
            level = log_level)
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

all_precisions = {}
all_recalls = {}
all_f1s = {}
for edited_llm in tqdm(llm_mapping.keys(), total=len(llm_mapping.keys())):
    all_precisions[edited_llm] = {}
    all_recalls[edited_llm] = {}
    all_f1s[edited_llm] = {}
    for edit_method in ['unke','ft', 'grace']:
        all_precisions[edited_llm][edit_method] = []
        all_recalls[edited_llm][edit_method] = []
        all_f1s[edited_llm][edit_method] = []

        LOG.info("Preprocess features..")
        test_feature_dir = f"{args.feature_dir}/{edit_method}_{llm_mapping[edited_llm]}_testset_token_6{'_rephrased' if args.rephrased else ''}.pkl"
        LOG.info(f"Load features from {test_feature_dir}.")
        if os.path.isfile(test_feature_dir):
            with open(test_feature_dir, 'rb') as f:
                test_feature = pickle.load(f)
        else:
            raise ValueError(f"File {test_feature_dir} isn't exsit.")
        train_feature_dir = f"{args.feature_dir}/{edit_method}_{llm_mapping[edited_llm]}_trainset_token_6{'_rephrased' if args.rephrased else ''}.pkl"
        LOG.info(f"Load features from {train_feature_dir}.")
        if os.path.isfile(train_feature_dir):
            with open(train_feature_dir, 'rb') as f:
                train_feature = pickle.load(f)
        else:
            raise ValueError(f"File {train_feature_dir} isn't exsit.")
        LOG.info("Load features done.")
        all_test_tokens_log_probs = test_feature['top_k_tokens_log_probs']
        all_train_tokens_log_probs = train_feature['top_k_tokens_log_probs']
        edited_llm_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path + edited_llm)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path + 'bert-base-uncased')
        for token_num in range(1, 7):
            torch.manual_seed(42)
            generated_texts_test, top20_probs_test = preprocess(all_test_tokens_log_probs, edited_llm_tokenizer, token_num)
            feature_test = ClosedLLMFeatureDataset(query=test_queries, generated_texts=generated_texts_test, top20_probs=top20_probs_test,
                                                labels=test_labels, tokenizer=tokenizer)
            generated_texts, top20_probs = preprocess(all_train_tokens_log_probs, edited_llm_tokenizer, token_num)
            feature_train = ClosedLLMFeatureDataset(query=train_queries, generated_texts=generated_texts, top20_probs=top20_probs,
                                                    labels=train_labels, tokenizer=tokenizer)

            train_dataloader = DataLoader(feature_train, batch_size=8, shuffle=True, worker_init_fn=np.random.seed(42))
            test_dataloader = DataLoader(feature_test, batch_size=8, shuffle=False, worker_init_fn=np.random.seed(42))

            LOG.info("Preprocess done.")

            model = Bert_RNNBasedClassifier(args.pretrained_model_path + 'bert-base-uncased', rnn_hidden_dim=256, num_classes=6).to(device)
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            LOG.info("Train start.")
            train_model(model, train_dataloader, criterion, optimizer, num_epochs=6)

            LOG.info("Train done.")
            LOG.info("Start eval.")
            model.eval()
            with torch.no_grad():
                accuracy, precision, recall, f1, cm = evaluate_model(model, test_dataloader)
            
            all_precisions[edited_llm][edit_method].append(precision)
            all_recalls[edited_llm][edit_method].append(recall)
            all_f1s[edited_llm][edit_method].append(f1)
            
            eval_res = f"{edited_llm}-{edit_method}-{token_num}\n" + f'Accuracy: {accuracy:.3f}\n' + f'Precision: {precision:.3f}\n' + f'Recall: {recall:.3f}\n' + f'F1 Score: {f1:.3f}'
            LOG.info(eval_res)
            
            plot_confusion_matrix(cm, type_mapping.keys(), f"./figs/ablation/cm_{edited_llm}_{edit_method}_tokens_{token_num}{'_rephrased' if args.rephrased else ''}.pdf")
            
            del model
            torch.cuda.empty_cache()
            gc.collect()

            del generated_texts_test, top20_probs_test, feature_test
            del generated_texts, top20_probs, feature_train
            torch.cuda.empty_cache()
            gc.collect()
            

allres = {"Precision": all_precisions, "Recall": all_recalls, "F1": all_f1s}
with open("./results/ablation_close_res.json", 'w') as f:
    json.dump(allres, f, indent=4)
