from easyeditor import BaseEditor, FTHyperParams, GraceHyperParams, UnkeHyperParams, NON_EDITHyperParams
import argparse
import logging
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys

sys.setrecursionlimit(10000) # FOR GRACE

LOG = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default='FT', type=str)
    parser.add_argument('--hparams_dir', default='./hparams/FT/llama2-13b', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--res_save_dir', default='./results', type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--log_level', default='INFO', type=str)

    args = parser.parse_args()
    log_level = logging.INFO
    if args.log_level == 'DEBUG':
        log_level = logging.DEBUG
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = log_level)
    
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'UNKE':
        editing_hparams = UnkeHyperParams
    elif args.editing_method == 'non-edit':
         editing_hparams = NON_EDITHyperParams
    else:
        raise NotImplementedError(f"Method {args.editing_method} is not implemented yet")

    hparams = editing_hparams.from_hparams(args.hparams_dir)

    datas = json.load(open(f'{args.data_dir}/edit_intention_all.json'))
    if args.ds_size:
        datas = datas[:args.ds_size]

    prompts, targets, rephrase_prompts = [], [], []
    for data in datas:
        if data['type']!= "Non edited":
            prompts.append(data['query'])
            targets.append(data['object'])
            rephrase_prompts.append(data['paraphrased_query'])


    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        ground_truth=None,
        target_new=targets,
        edit_batch=len(prompts),
        rephrase_prompts=rephrase_prompts
    )

    # Save the metrics
    os.makedirs(args.res_save_dir, exist_ok=True)
    res_path = f"{args.res_save_dir}/{args.editing_method.lower()}-{args.hparams_dir.split('/')[-1]}.json"
    LOG.info(f"Saving edited results to {res_path}")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save the edited model
    edited_model_path = f"./edited_model/{args.editing_method.lower()}/{args.hparams_dir.split('/')[-1]}"
    LOG.info(f"Saving edited model to {edited_model_path}")
    os.makedirs(edited_model_path, exist_ok=True)
    edited_model.save_pretrained(edited_model_path)
