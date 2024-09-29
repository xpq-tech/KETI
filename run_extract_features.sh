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
edit_methods=("ft" "grace" "unke")
edited_llms=("llama3.1-8b" "llama2-13b")

# 遍历所有组合
for edit_method in "${edit_methods[@]}"; do
    for edited_llm in "${edited_llms[@]}"; do
        if [ "$use_rephrased" == "true" ]; then
            # 打印当前组合
            echo "执行组合： extract_features edit_method=$edit_method, edited_llm=$edited_llm, rephrased=$use_rephrased"
            
            # 执行 Python 程序
            CUDA_VISIBLE_DEVICES=$gpu_number python extract_features.py --edit_method "$edit_method" --edited_llm "$edited_llm"  --rephrased --all_hidden_states
        else
                            # 打印当前组合
            echo "extract_features edit_method=$edit_method, edited_llm=$edited_llm"
            
            # 执行 Python 程序
            CUDA_VISIBLE_DEVICES=$gpu_number python extract_features.py --edit_method "$edit_method" --edited_llm "$edited_llm" --all_hidden_states
        fi
    done
done