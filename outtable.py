from beautifultable import BeautifulTable
import os
import glob
import json

base_dir = 'results/quantitative'
methods = [x for x in os.listdir(base_dir)]
result_dir = {}

for method in methods:
    method_dir = os.path.join(base_dir, method)
    method = method[:15]
    latest_cknum = max([int(name.split("_")[0]) for name in os.listdir(method_dir) if name[:4] not in ['cont', 'coco']])
    filepaths = glob.glob(f"{method_dir}/{latest_cknum}*") + glob.glob(f"{method_dir}/cont*") + glob.glob(f"{method_dir}/cocon*")
    for fpath in filepaths:
        data = json.load(open(fpath))
        task = data['task']
        dataset = data['dataset']
        if os.path.basename(fpath).startswith('c'): show_method = "-".join(os.path.basename(fpath).split("_")[:3]) + "_" + method
        else: show_method = method
        
        if task == 'zeroshot_classification':
            try: metrics = data['metrics']['acc1']
            except: continue
            if task not in result_dir.keys(): result_dir[task] = {}
            if show_method not in result_dir[task].keys(): result_dir[task][show_method] = {}
            result_dir[task][show_method][dataset] = metrics
        elif task == 'compositionality':
            metrics = data['metrics']
            assert task == 'compositionality'
            data_task = task + "_" + dataset
            if data_task not in result_dir.keys(): result_dir[data_task] = {}
            result_dir[data_task][show_method] = metrics
        elif task == 'zeroshot_retrieval':
            metrics = data['metrics']
            if task not in result_dir.keys(): result_dir[task] = {}
            if show_method not in result_dir[task].keys(): result_dir[task][show_method] = {}
            result_dir[task][show_method][dataset + "_image"] = metrics["image_retrieval_recall@5"]
            result_dir[task][show_method][dataset + "_text"] = metrics["text_retrieval_recall@5"]
            
            

for task, method_result in result_dir.items():
    table = BeautifulTable(maxwidth=1000)
    row_header = []
    enforced_order = ['wds/imagenet1k', 'wds/vtab/cifar100', 'wds/vtab/cifar100', 'wds/stl10', 'wds/voc2007', 'wds/vtab/caltech101', 'wds/sun397', 'wds/vtab/pets', 'wds/vtab/flowers', 'wds/food101', 'wds/objectnet', 'wds/vtab/clevr_closest_object_distance', 'wds/vtab/smallnorb_label_azimuth','wds/vtab/resisc45', 'wds/vtab/dmlab', 'wds/imagenet-a', 'wds/imagenet-r', 'wds/imagenet_sketch', ]
    original_order = list(method_result[list(method_result.keys())[0]].keys())
    table.columns.header = enforced_order + [o for o in original_order if o not in enforced_order]
    for method, dataset_result in method_result.items():
        row_header.append(method)
        row = []
        for k in table.columns.header: 
            if k in dataset_result.keys(): row.append(dataset_result[k])
            else: row.append("-")
        table.rows.append(row)
    table.rows.header = row_header
    print(table)
        
