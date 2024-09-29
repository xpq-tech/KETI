from baselines import MLPClassifier, LinearClassifier
from easyeditor import KEIDDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
import argparse
import pickle
import logging
import os
import torch
torch.manual_seed(42)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
type_mapping = {
    "Non edited": 0,
    "Fact updating": 1,
    "Misinformation injection": 2,
    "Offensiveness injection": 3,
    "Behavioral misleading injection": 4,
    "Bias injection": 5
}
LOG = logging.getLogger(__name__)

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
    parser.add_argument('--method', default='MLP', choices=['linear', 'MLP', 'LDA', 'LogR'], type=str)
    parser.add_argument('--edit_method', default='ft', choices=['ft','grace','unke'], type=str)
    parser.add_argument('--edited_llm', default='llama3.1-8b',  choices=['llama3.1-8b','llama2-13b'], type=str)
    parser.add_argument('--feature_dir', default='./features', type=str)
    parser.add_argument('--rephrased', default=False, action="store_true")
    parser.add_argument('--test_feature', default=None,  choices=['ft','grace','unke', 'non-edit'])
    parser.add_argument('--more_tokens', default=6, type=int)
    parser.add_argument('--log_level', default='INFO', type=str)

    args = parser.parse_args()

    log_level = logging.INFO
    if args.log_level == 'DEBUG':
        log_level = logging.DEBUG
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = log_level)

    test_set = KEIDDataset('./data/edit_intention_split/test.json', type_mapping)
    train_set = KEIDDataset('./data/edit_intention_split/train.json', type_mapping)

    LOG.info("Preprocess features..")
    if args.test_feature is not None:
        test_feature_dir = f"{args.feature_dir}/{args.test_feature}_{args.edited_llm}_testset_token_{args.more_tokens}{'_rephrased' if args.rephrased and args.test_feature!='non-edit' else ''}.pkl"
    else:
        test_feature_dir = f"{args.feature_dir}/{args.edit_method}_{args.edited_llm}_testset_token_{args.more_tokens}{'_rephrased' if args.rephrased else ''}.pkl"
    if os.path.isfile(test_feature_dir):
        with open(test_feature_dir, 'rb') as f:
            test_feature = pickle.load(f)
    else:
        raise ValueError(f"File {test_feature_dir} isn't exsit.")
    train_feature_dir = f"{args.feature_dir}/{args.edit_method}_{args.edited_llm}_trainset_token_{args.more_tokens}{'_rephrased' if args.rephrased else ''}.pkl"
    if os.path.isfile(train_feature_dir):
        with open(train_feature_dir, 'rb') as f:
            train_feature = pickle.load(f)
    else:
        raise ValueError(f"File {train_feature_dir} isn't exsit.")


    all_test_hs = test_feature['hidden_states']
    all_train_hs = train_feature['hidden_states']

    if args.method in ["linear", "MLP"]:
    # 特征归一化
        last_train_hs = []
        for hs in all_train_hs:
            last_train_hs.append(hs[-1])
        last_train_hs = torch.stack(last_train_hs)
        last_train_hs = (last_train_hs - last_train_hs.min()) / (last_train_hs.max() - last_train_hs.min())
        last_test_hs = []
        for hs in all_test_hs:
            last_test_hs.append(hs[-1])
        last_test_hs = torch.stack(last_test_hs)
        last_test_hs = (last_test_hs - last_test_hs.min()) / (last_test_hs.max() - last_test_hs.min())
        train_labels = torch.tensor([label for _,_, label in train_set])
        if args.test_feature is not None and args.test_feature == 'non-edit':
            test_labels = torch.tensor([label for _,_, label in test_set]) * 0
        else:
            test_labels = torch.tensor([label for _,_, label in test_set])
        train_dataset = TensorDataset(last_train_hs.float(), train_labels)
        test_dataset = TensorDataset(last_test_hs.float(), test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        LOG.info("Preprocess features done.")


        input_dim = last_test_hs.shape[-1]
        num_classes = 6
        learning_rate = 1e-3
        epochs = 2000

        if args.method == "linear":
            model = LinearClassifier(input_dim, num_classes).to(device)
        elif args.method == "MLP":
            model = MLPClassifier(input_dim, num_classes, hidden_dim1=256, hidden_dim2=16).to(device)
        else:
            raise NotImplementedError(f"Method {args.method} not implement yet!")
        # 评估模型
        LOG.info(f"Use {args.method} classifier")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 训练
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            LOG.info(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}')
        LOG.info("Train done.")
        LOG.info("Start eval.")
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x, y in test_dataloader:
                # 确保输入的数据类型与模型一致
                x = x.float().to(device)
                y = y.long().to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        testset_labels = all_labels
        predictions = all_preds
    else:
        if args.test_feature is not None and args.test_feature == 'non-edit':
            testset_labels = np.array([label for _,_, label in test_set])*0
        else:
            testset_labels = np.array([label for _,_, label in test_set])
        np_train_hs = []
        for hs in all_train_hs:
            np_train_hs.append(hs[-1].numpy())
        np_train_hs = np.array(np_train_hs)

        np_test_hs = []
        for hs in all_test_hs:
            np_test_hs.append(hs[-1].numpy())
        np_test_hs = np.array(np_test_hs)

        
        y = np.array([label for _,_, label in train_set])

        if args.method == "LDA":
            model = LDA()
        elif args.method == "LogR":
            model = LogisticRegression( solver='lbfgs', max_iter=1000)
        else:
            raise NotImplementedError(f"{args.method} not implemented")
        LOG.info(f"Use {args.method} classifier")
        model.fit(np_train_hs, y)
        predictions = model.predict(np_test_hs)

    # 计算分类准确率
    
    total = len(testset_labels)
    correct = sum(p == l for p, l in zip(predictions, testset_labels))
    accuracy = correct / total

    # 计算 Precision, Recall 和 F1 Score
    precision = precision_score(testset_labels, predictions, average='macro')
    recall = recall_score(testset_labels, predictions, average='macro')
    f1 = f1_score(testset_labels, predictions, average='macro')

    conf_matrix = confusion_matrix(testset_labels, predictions)
    eval_res = f'Accuracy: {accuracy:.3f}\n' + f'Precision: {precision:.3f}\n' + f'Recall: {recall:.3f}\n' + f'F1 Score: {f1:.3f}' + f'Confusion Matrix:\n{conf_matrix}'
    with open(f"./results/{args.method}_{args.edit_method}{f'_to_{args.test_feature}' if args.test_feature is not None else ''}_{args.edited_llm}{'_rephrased' if args.rephrased else ''}.txt", 'w') as file:  # 'w' 模式会覆盖文件内容，'a' 模式是追加内容
        file.write(eval_res)
    print(eval_res)
    # 2. 输出混淆矩阵
    plot_confusion_matrix(conf_matrix, ['NE', "FU", "MI", "OI", "BMI", "BI"], f"./figs/cm_{args.method}_{args.edit_method}{f'_to_{args.test_feature}' if args.test_feature is not None else ''}_{args.edited_llm}{'_rephrased' if args.rephrased else ''}.pdf")
