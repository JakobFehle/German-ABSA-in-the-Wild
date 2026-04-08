import subprocess
import sys
import os
import pandas as pd
import numpy as np

MODEL_NAME = "meta-llama/Llama-3.1-8B"
LORA_DROPOUT = 0.05
QLORA_QUANT = 4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LR_SCHEDULER = 'linear'
LORA_R, LORA_ALPHA = 64, 16
LEARNING_RATE = 2e-4
OUTPUT_DIR = '../results/ft_llm/'
DATA_SETTING = 'orig'


###
# Hyperparameter Validation Phase
###

EVAL_TYPE = 'dev'
SEED = 5

DATASETS_EPOCHS = [
            ['restaurant', [4, 5, 6, 7]], 
            ['inclusion', [4, 5, 6, 7]],
            ['transport', [1]],
            ['software/v2', [4, 5, 6, 7]],
            ['hotel', [4, 5, 6, 7]]
            ]

for TASK in ['acd', 'acsa', 'tasd']:
    DATASETS_EPOCHS = DATASETS_EPOCHS[:-1] if (TASK == 'tasd' and 'hotel' in [DATASET[0] for DATASET in DATASETS_EPOCHS]) else DATASETS_EPOCHS
    for DATASET, EPOCH_RANGE in DATASETS_EPOCHS:
        for EPOCHS in EPOCH_RANGE:
                    
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 ../src/ft_llm/train.py \
            --model_name_or_path {MODEL_NAME} \
            --lora_r {LORA_R} \
            --lora_alpha {LORA_ALPHA} \
            --lora_dropout {LORA_DROPOUT} \
            --quant {QLORA_QUANT} \
            --eval_type {EVAL_TYPE} \
            --learning_rate {LEARNING_RATE} \
            --per_device_train_batch_size {BATCH_SIZE} \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS} \
            --num_train_epochs {EPOCHS} \
            --dataset {DATASET} \
            --data_setting {DATA_SETTING} \
            --task {TASK} \
            --lr_scheduler {LR_SCHEDULER} \
            --bf16 \
            --group_by_length \
            --output_dir {OUTPUT_DIR} \
            --seed {SEED} \
            --flash_attention"
    
            process = subprocess.Popen(command, shell=True)
            process.wait()



col_names = ['task', 'dataset', 'eval_type', 'data_setting', 'learning_rate', 'batch_size', 'epoch', 'seed', 'f1-micro', 'f1-macro', 'accuracy']
runs = []

folder_names = [folder for folder in os.listdir(os.path.join(OUTPUT_DIR)) if os.path.isdir(os.path.join(OUTPUT_DIR, folder)) and folder != '.ipynb_checkpoints']

for folder_name in folder_names:
    try:
        cond_parameters = folder_name.split('_')

        if cond_parameters[0] == 'acd':
            df = pd.read_csv(os.path.join(OUTPUT_DIR, folder_name, 'metrics_asp.tsv'), sep = '\t')
        elif cond_parameters[0] == 'acsa':
            df = pd.read_csv(os.path.join(OUTPUT_DIR, folder_name, 'metrics_asp_pol.tsv'), sep = '\t')
        elif cond_parameters[0] == 'tasd':
            df = pd.read_csv(os.path.join(OUTPUT_DIR, folder_name, 'metrics_phrases.tsv'), sep = '\t')
        df = df.set_index(df.columns[0])
        
        cond_parameters.append(df.loc['Micro-AVG', 'f1'])
        cond_parameters.append(df.loc['Macro-AVG', 'f1'])
        cond_parameters.append(df.loc['Micro-AVG', 'accuracy'])
        runs.append(cond_parameters)
    except:
        pass

results_all = pd.DataFrame(runs, columns = col_names)
results_all["f1-micro"] = pd.to_numeric(results_all["f1-micro"], errors="coerce")

##
# Test Evaluation Phase
##

EVAL_TYPE = 'test'
for TASK in ['tasd', 'acd', 'acsa']:
    for SEED in [5, 10, 15, 20, 25]:        
        for DATA_SETTING in ['orig']:
            for index, DATASET in enumerate([DATASET[0] for DATASET in DATASETS_EPOCHS]):
    
                results_sub = results_all[np.logical_and.reduce([results_all['data_setting'] == f'{DATA_SETTING}-{DATA_SETTING[0]}', 
                                                            results_all['dataset'] == DATASET.replace('/','-'),
                                                            results_all['task'] == TASK,
                                                            results_all['eval_type'] == 'dev'])].sort_values(by = ['f1-micro'], ascending = False)
                results_sub = results_sub.reset_index()
    
                print(results_sub.head(3))
                EPOCHS = int(results_sub.at[0, 'epoch'])
    
                command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 ../src/ft_llm/train.py \
                --model_name_or_path {MODEL_NAME} \
                --lora_r {LORA_R} \
                --lora_alpha {LORA_ALPHA} \
                --lora_dropout {LORA_DROPOUT} \
                --quant {QLORA_QUANT} \
                --eval_type {EVAL_TYPE} \
                --learning_rate {LEARNING_RATE} \
                --per_device_train_batch_size {BATCH_SIZE} \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS} \
                --num_train_epochs {EPOCHS} \
                --dataset {DATASET} \
                --data_setting {DATA_SETTING} \
                --task {TASK} \
                --lr_scheduler {LR_SCHEDULER} \
                --bf16 \
                --group_by_length \
                --output_dir {OUTPUT_DIR} \
                --seed {SEED} \
                --flash_attention"
        
                process = subprocess.Popen(command, shell=True)
                process.wait()