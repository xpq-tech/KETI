import json
import argparse
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', default='./results', type=str)
    args = parser.parse_args()
    # 读取 JSON 文件
    for json_file in glob.glob(os.path.join(args.res_dir, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
    
        # 初始化累加器
        total_rewrite_acc = 0
        total_rephrase_acc = 0
        rewrite_count = 0
        rephrase_count = 0

        # 遍历数据列表
        for item in data:
            # 计算 rewrite_acc 的总和和数量
            rewrite_acc_list = item.get('post', {}).get('rewrite_acc', [])
            total_rewrite_acc += sum(rewrite_acc_list)
            rewrite_count += len(rewrite_acc_list)
            
            # 计算 rephrase_acc 的总和和数量
            rephrase_acc_list = item.get('post', {}).get('rephrase_acc', [])
            total_rephrase_acc += sum(rephrase_acc_list)
            rephrase_count += len(rephrase_acc_list)

        # 计算平均值
        rewrite_acc_mean = total_rewrite_acc / rewrite_count if rewrite_count > 0 else 0
        rephrase_acc_mean = total_rephrase_acc / rephrase_count if rephrase_count > 0 else 0

        # 打印结果
        print("+"*100)
        print(f"Results in: {json_file}")
        print(f"Average rewrite_acc: {rewrite_acc_mean}")
        print(f"Average rephrase_acc: {rephrase_acc_mean}")
