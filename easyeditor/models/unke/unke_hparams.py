from dataclasses import dataclass
from typing import List
from ...util.hparams import HyperParams
import yaml


@dataclass
class UnkeHyperParams(HyperParams):
    # Experiments
    model_name: str #'Qwen1.5-7B-Chat' 
    keep_original_weight: bool
    alg_name: str
    ex_data_path: str


    lr: str
    batch_size: str
    layers: str
    ln_f_module: str
    lm_head_module: str
    layer_module_tmp: str
    #rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

    device: int

    v_loss_layer: int
    v_lr: float
    v_num_grad_steps: int
    v_weight_decay: float
    clamp_norm_factor: int
    optim_num_step: int
    ex_data_num: int

    debug: bool = False
    model_parallel: bool = False
   

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'UNKE') or print(
            f'UnkeHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
