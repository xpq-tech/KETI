from easyeditor import BaseEditor, FTHyperParams, GraceHyperParams, KEIDDataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
torch.manual_seed(42)
import pickle
import argparse


type_mapping = {
    "Non edited": 0,
    "Fact updating": 1,
    "Misinformation injection": 2,
    "Offensiveness injection": 3,
    "Behavioral misleading injection": 4,
    "Bias injection": 5
}

def get_features(prompt, more_tokens=5, top_k=1):
    inp_tok = tokenizer(prompt, return_tensors="pt",add_special_tokens=False,).to(edited_model.device)

    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)
    max_out_len = more_tokens + input_ids.size(1)
    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())
    max_logits, top_k_tokens_log_probs = [], []
    last_layer_hs = None
    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = edited_model(
                input_ids=input_ids[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            if last_layer_hs is None:
                hidden_states = model_out.hidden_states
                # 返回最后一层的hidden states
                last_layer_hs = hidden_states[-1].detach().cpu()
            
            logits, past_key_values = model_out.logits, model_out.past_key_values
            log_probs = torch.log_softmax(logits[:,-1,:], dim=-1)
            top_k_log_probs, top_k_ids = torch.topk(log_probs, 20, dim=-1)
            top_k_tokens_log_probs.append({
                    "token_ids": top_k_ids[0].cpu(),
                    "logprobs": top_k_log_probs[0].cpu()})
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)
            max_logits.append(logits.max().cpu())
            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tokenizer.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)
        return max_logits, last_layer_hs, top_k_tokens_log_probs

def get_features_all_hs(prompt):
    inp_tok = tokenizer(prompt, return_tensors="pt",add_special_tokens=False,).to(edited_model.device)
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())
    with torch.no_grad():
        model_out = edited_model(
            input_ids=input_ids[:, cur_context],
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        hidden_states = model_out.hidden_states
        all_hs = [hs.detach().cpu() for hs in hidden_states]
        return all_hs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--edit_method', default='ft',choices=['ft','grace','unke', 'non-edit'], type=str)
    parser.add_argument('--edited_llm', default='llama3.1-8b', choices=['llama3.1-8b','llama2-13b'], type=str)
    parser.add_argument('--feature_dir', default='./features', type=str)
    parser.add_argument('--more_tokens', default=6, type=int)
    parser.add_argument('--rephrased', default=False, action="store_true")
    parser.add_argument('--log_level', default='INFO', type=str)
    parser.add_argument('--edited_model_dir', default='./edited_model', type=str)
    parser.add_argument('--pretrained_model_path', default="/science/llms/", type=str)
    parser.add_argument('--all_hidden_states', default=True, action="store_true")


    args = parser.parse_args()

    test_set = KEIDDataset('./data/edit_intention_split/test.json', type_mapping)
    train_set = KEIDDataset('./data/edit_intention_split/train.json', type_mapping)

    datas = json.load(open('./data/edit_intention_all.json'))

    if args.edited_llm == "llama3.1-8b":
        tokenizer = AutoTokenizer.from_pretrained(f"{args.pretrained_model_path}/Meta-Llama-3.1-8B-Instruct")
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"{args.pretrained_model_path}/Llama-2-13b-chat-hf")

    tokenizer.pad_token = tokenizer.eos_token

    # FT-M UNKE
    if args.edit_method in ['ft', 'unke']:
        model_path = f"{args.edited_model_dir}/{args.edit_method}/{args.edited_llm}"
        print(f"Loading model from {model_path}")
        edited_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    elif args.edit_method == 'grace':
        #GRACE
        hparams = GraceHyperParams.from_hparams(f'./hparams/GRACE/{args.edited_llm}.yaml')
        hparams.adapter_path = f"{args.edited_model_dir}/{args.edit_method}/{args.edited_llm}/{hparams.inner_params[0][:-7]}.pt"
        print(f"Loading GRACE adapter from {hparams.adapter_path}")
        editor = BaseEditor.from_hparams(hparams)
        edited_model, weights_copy = editor.apply_algo(editor.model,editor.tok,[],hparams)
    else:
        if args.edited_llm == "llama3.1-8b":
            edited_model = AutoModelForCausalLM.from_pretrained(f"{args.pretrained_model_path}/Meta-Llama-3.1-8B-Instruct").cuda()
        else:
            edited_model = AutoModelForCausalLM.from_pretrained(f"{args.pretrained_model_path}/Llama-2-13b-chat-hf").cuda()

    edit_method = args.edit_method
    edited_llms = args.edited_llm
    more_tokens = args.more_tokens

    for item in ['testset', 'trainset']:
        if item == 'testset':
            datas = test_set
        else:    
            datas = train_set
        all_hs = []
        if args.all_hidden_states:
            for prompt, rephrased_prompt, label in tqdm(datas, total=len(datas)):
                if args.rephrased:
                    hs = get_features_all_hs(rephrased_prompt)
                else:
                    hs = get_features_all_hs(prompt)
                all_hs.append(hs)
            features = {'all_hidden_states':all_hs}
            if args.rephrased:
                data_file = f'{args.feature_dir}/all_hs/{edit_method}_{edited_llms}_{item}_rephrased.pkl'
            else:
                data_file = f'{args.feature_dir}/all_hs/{edit_method}_{edited_llms}_{item}.pkl'
            with open(data_file, 'wb') as f:
                pickle.dump(features, f)
        else:
            all_max_logits = []
            all_top_k_tokens_log_probs = []
            for prompt, rephrased_prompt, label in tqdm(datas, total=len(datas)):
                if args.rephrased:
                    prompt_max_logits, prompt_last_layer_hs, top_k_tokens_log_probs = get_features(rephrased_prompt, more_tokens=more_tokens)
                else:
                    prompt_max_logits, prompt_last_layer_hs, top_k_tokens_log_probs = get_features(prompt, more_tokens=more_tokens)
                all_max_logits.extend(prompt_max_logits)
                all_hs.extend(prompt_last_layer_hs)
                all_top_k_tokens_log_probs.append(top_k_tokens_log_probs)

            features = {'max_logits': all_max_logits, 'hidden_states':all_hs, 'top_k_tokens_log_probs': all_top_k_tokens_log_probs}
            if args.rephrased:
                data_file = f'{args.feature_dir}/{edit_method}_{edited_llms}_{item}_token_{more_tokens}_rephrased.pkl'
            else:
                data_file = f'{args.feature_dir}/{edit_method}_{edited_llms}_{item}_token_{more_tokens}.pkl'
            with open(data_file, 'wb') as f:
                pickle.dump(features, f)



