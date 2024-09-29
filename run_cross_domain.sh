#!/bin/bash
usage() {
    echo "Usage: $0 <gpu_number> <use_rephrased>"
    echo "<gpu_number>: Integer specifying the GPU index."
    echo "<use_rephrased>: Boolean (true/false) to specify whether to use the 'rephrased' parameter."
    exit 1
}

# 检查参数个数
if [ "$#" -ne 2 ]; then
    usage
fi

# 解析输入参数
gpu_number=$1
use_rephrased=$2

# 检查显卡编号是否为整数
if ! [[ "$gpu_number" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU编号必须是一个整数。"
    usage
fi

# 检查布尔参数是否有效
if [[ "$use_rephrased" != "true" && "$use_rephrased" != "false" ]]; then
    echo "Error: use_rephrased 参数必须是 'true' 或 'false'"
    usage
fi

# 定义所有参数的取值范围
# methods=("linear" "MLP" "LDA" "LogR")
methods=("LogR")
edit_methods=("ft" "grace" "unke")
edited_llms=("llama3.1-8b" "llama2-13b")
test_features=("ft" "grace" "unke" "non-edit")

# 遍历所有组合
for method in "${methods[@]}"; do
    for edit_method in "${edit_methods[@]}"; do
        for edited_llm in "${edited_llms[@]}"; do
            for test_feature in "${test_features[@]}"; do
                # 跳过 edit_method 和 test_feature 相同的情况
                if [ "$edit_method" == "$test_feature" ]; then
                    echo "跳过执行：edit_method ($edit_method) 和 test_feature ($test_feature) 相同。"
                    continue
                fi
                if [ "$use_rephrased" == "true" ]; then
                # 打印当前组合
                    echo "执行组合：open_source_llms_main method=$method, edit_method=$edit_method, edited_llm=$edited_llm, test_feature=$test_feature, use_rephrased=$use_rephrased"
                    
                    # 执行 Python 程序
                    CUDA_VISIBLE_DEVICES=$gpu_number python open_source_llms_main.py --method "$method" --edit_method "$edit_method" --edited_llm "$edited_llm" --test_feature "$test_feature" --rephrased
                else
                # 打印当前组合
                    echo "执行组合：open_source_llms_main method=$method, edit_method=$edit_method, edited_llm=$edited_llm, test_feature=$test_feature, use_rephrased=$use_rephrased"
                    
                    # 执行 Python 程序
                    CUDA_VISIBLE_DEVICES=$gpu_number python open_source_llms_main.py --method "$method" --edit_method "$edit_method" --edited_llm "$edited_llm" --test_feature "$test_feature"
                fi
            done
        done
    done
done


methods=("BERT+LSTM")
edit_methods=("ft" "grace" "unke")
edited_llms=("Meta-Llama-3.1-8B-Instruct" "Llama-2-13b-chat-hf")
test_features=("ft" "grace" "unke" "non-edit")

# 遍历所有组合
for method in "${methods[@]}"; do
    for edit_method in "${edit_methods[@]}"; do
        for edited_llm in "${edited_llms[@]}"; do
            for test_feature in "${test_features[@]}"; do
                # 跳过 edit_method 和 test_feature 相同的情况
                if [ "$edit_method" == "$test_feature" ]; then
                    echo "跳过执行：close_source_llms_main edit_method ($edit_method) 和 test_feature ($test_feature) 相同。"
                    continue
                fi
                if [ "$use_rephrased" == "true" ]; then
                    echo "执行组合：close_source_llms_main method=$method, edit_method=$edit_method, edited_llm=$edited_llm, test_feature=$test_feature, use_rephrased=$use_rephrased"
                    
                    # 执行 Python 程序
                    CUDA_VISIBLE_DEVICES=$gpu_number python close_source_llms_main.py --method "$method" --edit_method "$edit_method" --edited_llm "$edited_llm" --test_feature "$test_feature" --rephrased
                else
                    # 打印当前组合
                    echo "执行组合： close_source_llms_main method=$method, edit_method=$edit_method, edited_llm=$edited_llm, test_feature=$test_feature, use_rephrased=$use_rephrased"
                    
                    # 执行 Python 程序
                    CUDA_VISIBLE_DEVICES=$gpu_number python close_source_llms_main.py --method "$method" --edit_method "$edit_method" --edited_llm "$edited_llm" --test_feature "$test_feature" 
                fi
            done
        done
    done
done