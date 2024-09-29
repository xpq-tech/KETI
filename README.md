# Knowledge Editing Intention Detection (KEID)

## Requirements

#### 🔧 Installation via Pip and Conda

**Note: Please use Python 3.9+ for KEID**

To get started, install Conda and run the following:

```shell
conda create -n keti python=3.10
...
pip install -r requirements.txt
```

## How to Reproduce This Study

### 1. Prepare Llama 3.1-8B-Instruct and Llama 2-13B-Chat LLMs locally. Then, update their paths in the **model_name** field in [hparams](./hparams/) for the knowledge editing methods.

### 2. Use the [edit_llms.sh](./edit_llms.sh) script to perform knowledge editing. Once complete, summarize the results using the following command:

```shell
python summarize_edit_results.py [results_file_path]
```

### 3. Run [run_extract_features.sh](./run_extract_features.sh) to extract features from the edited LLMs:

```shell
bash run_extract_features.sh [GPU_Number] [default: false; set to true to extract features from rephrased queries]
```

### 4. Run [run_all_baselines.sh](./run_all_baselines.sh) to perform all knowledge editing type identification experiments using the baseline identifiers.
```shell
bash run_all_baselines.sh [GPU_Number] [default: false; set to true to extract features from rephrased queries]
```

### 5. Run [run_cross_domain.sh](./run_cross_domain.sh) to execute cross-domain identification experiments.
```shell
bash run_cross_domain.sh [GPU_Number] [default: false; set to true to extract features from rephrased queries]
```

### Ablation Studies

Refer to the notebooks for closed-source and open-source LLMs:

- [ablation_closed_llm.ipynb](./ablation_closed_llm.ipynb)
- [ablation_open_llm.ipynb](./ablation_open_llm.ipynb)

### Additional Analyses

Refer to [analysis.ipynb](./analysis.ipynb) for other analyses related to the experiments.

## Acknowledgement

This project builds upon [EasyEdit](https://github.com/zjunlp/EasyEdit).