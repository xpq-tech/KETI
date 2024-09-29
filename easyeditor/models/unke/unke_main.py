from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from .unke import execute_batch_unke, get_llama_without_answer
from .unke_hparams import UnkeHyperParams
from ...util import nethook
import json
import random

def apply_unke_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: UnkeHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    
    with open(hparams.ex_data_path, 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
    ex_datas = [get_llama_without_answer(i['instruction']+i['input'])+i['output']  for i in ex_datas]
    
    random_elements = random.sample(ex_datas, hparams.ex_data_num)
    weights_copy = execute_batch_unke(model, tok, hparams, requests, random_elements)


    return model, weights_copy


