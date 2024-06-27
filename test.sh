#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 model_name ckpt_path task"
    exit 1
fi

MODEL_NAME=$1
CKPT_PATH=$2
TASK=$3


# Generate EPOCH_LIST starting from 99, decreasing by 3, until reaching 3
EPOCH_LIST=$(seq 100000 -1 1)



for ITER in ${EPOCH_LIST}; do
    path_to_check=${CKPT_PATH}/ckpt_${ITER}.pth.tar
    
    if [ -e "$path_to_check" ]; then
        echo ${path_to_check}
        output_folder=${CKPT_PATH}/results
        mkdir $output_folder
        
        # Conditional execution based on the TASK argument
        if [ "$TASK" == "probing" ]; then
            echo "Executing probing Commands"
            python CLIP_benchmark/clip_benchmark/cli.py eval \
                --model_type cust_clip \
                --model $MODEL_NAME \
                --pretrained $path_to_check \
                --dataset webdatasets.txt \
                --wds_cache_dir /gscratch/krishna/chenhaoz/IL/FDT/data \
                --dataset_root https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main \
                --output $output_folder/${ITER}_{dataset}_{task}.json \
                --task linear_probe \
                --feature_root tmp_features \
                --fewshot_epochs 50 \
                --fewshot_lr 0.1 \
                --batch_size 256

        # Conditional execution based on the TASK argument
        elif [ "$TASK" == "classification" ]; then
            echo "Executing Classification Commands"
            python CLIP_benchmark/clip_benchmark/cli.py eval \
                --model_type cust_clip \
                --model $MODEL_NAME \
                --pretrained $path_to_check \
                --dataset webdatasets.txt \
                --wds_cache_dir /gscratch/krishna/chenhaoz/IL/FDT/data \
                --dataset_root https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main \
                --output $output_folder/${ITER}_{dataset}_{task}.json \
                --task zeroshot_classification

        # Conditional execution based on the TASK argument
        elif [ "$TASK" == "retrieval" ]; then
            echo "Executing Retrieval Commands"
            python CLIP_benchmark/clip_benchmark/cli.py eval \
                --model_type cust_clip \
                --model $MODEL_NAME \
                --pretrained $path_to_check \
                --dataset webdatasets.txt \
                --wds_cache_dir /gscratch/krishna/chenhaoz/IL/FDT/data \
                --dataset_root https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main \
                --output $output_folder/${ITER}_{dataset}_{task}.json \
                --task zeroshot_retrieval

        elif [ "$TASK" == "compositionality" ]; then
            echo "Executing Compositionality Commands"
            python CLIP_benchmark/clip_benchmark/eval/sugar_crepe.py \
                --model $MODEL_NAME \
                --pretrained $path_to_check \
                --output_folder $output_folder \
                --iter ${ITER}

            python CLIP_benchmark/clip_benchmark/eval/cola_multi.py \
                --model $MODEL_NAME \
                --pretrained $path_to_check \
                --output_folder $output_folder \
                --iter ${ITER}

            python CLIP_benchmark/clip_benchmark/eval/winoground.py \
                --model $MODEL_NAME \
                --pretrained $path_to_check \
                --output_folder $output_folder \
                --iter ${ITER}

            python CLIP_benchmark/clip_benchmark/eval/crepe.py \
                --model $MODEL_NAME \
                --pretrained $path_to_check \
                --output_folder $output_folder \
                --iter ${ITER}
        
        fi
    fi
done