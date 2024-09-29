from torch.utils.data import DataLoader, Dataset
import torch


class ClosedLLMFeatureDataset(Dataset):
    def __init__(self, query, generated_texts, top20_probs, labels, tokenizer, max_length=512):
        self.query = query
        self.generated_texts = generated_texts
        self.top20_probs = top20_probs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        text = self.query[idx]
        generated_text = self.generated_texts[idx]
        top20_prob = self.top20_probs[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        generated_inputs = self.tokenizer(generated_text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'generated_input_ids': generated_inputs['input_ids'].squeeze(0),
            'generated_attention_mask': generated_inputs['attention_mask'].squeeze(0),
            'top20_probs': top20_prob,
            'label': torch.tensor(label, dtype=torch.long)
        }