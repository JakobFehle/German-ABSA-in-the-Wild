import subprocess
import sys
import os
import pandas as pd
import numpy as np

# CONSTANTS

# MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MODEL_NAME = "meta-llama/Llama-3.1-8B-instruct"
QLORA_QUANT = 4
OUTPUT_DIR = '../results/fs_llm/'
DATA_SETTING = 'orig'
MAX_SEQ_LENGTH = 2048

###
# Hyperparameter Validation Phase
###

DATASETS_ALL = [
    'inclusion',
    'transport',
    'restaurant',
    'software/v2-l',
    'hotel',
]

EVAL_TYPE = 'dev'
SEED = 5

for TASK in ['acd','acsa', 'tasd']:
    DATASETS = DATASETS_ALL[:-1] if TASK == 'tasd' else DATASETS_ALL
    for DATASET in DATASETS:
        for FEW_SHOTS in [10, 25, 50]:
            
            if FEW_SHOTS == 10:
                SEQ_LEN = 2048
            elif FEW_SHOTS == 25:
                SEQ_LEN = 4096
            elif FEW_SHOTS == 50:
                SEQ_LEN = 8192
                
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 ../src/fs_llm/eval-vllm.py \
            --model_name_or_path {MODEL_NAME} \
            --quant {QLORA_QUANT} \
            --eval_type {EVAL_TYPE} \
            --dataset {DATASET} \
            --data_setting {DATA_SETTING} \
            --few_shots {FEW_SHOTS} \
            --task {TASK} \
            --bf16 \
            --max_seq_length {SEQ_LEN} \
            --output_dir {OUTPUT_DIR} \
            --seed {SEED} "
    
            process = subprocess.Popen(command, shell=True)
            process.wait()


RESULTS_PATH = '../results/fs_llm/'
col_names = ['task', 'dataset', 'eval_type', 'data_setting', 'few_shots', 'seed', 'f1-micro', 'f1-macro', 'accuracy']
runs = []

folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH)) if os.path.isdir(os.path.join(RESULTS_PATH, folder)) and folder != '.ipynb_checkpoints']

for folder_name in folder_names:
    try:
        cond_parameters = folder_name.split('_')

        if cond_parameters[0] == 'acd':
            df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_asp.tsv'), sep = '\t')
        elif cond_parameters[0] == 'acsa':
            df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_asp_pol.tsv'), sep = '\t')
        elif cond_parameters[0] == 'tasd':
            df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_phrases.tsv'), sep = '\t')
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
for TASK in ['acd','acsa', 'tasd']:
    for DATASET in DATASETS:
        for SEED in [5,10,15,20,25]:        
            for DATA_SETTING in ['orig']:
                results_sub = results_all[np.logical_and.reduce([results_all['data_setting'] == f'{DATA_SETTING}-{DATA_SETTING[0]}', 
                                                            results_all['dataset'] == DATASET.replace('/','-'),
                                                            results_all['task'] == TASK,
                                                            results_all['eval_type'] == 'dev'])].sort_values(by = ['f1-micro'], ascending = False)
                results_sub = results_sub.reset_index()
    
                print(results_sub.head(3))
                FEW_SHOTS = int(results_sub.at[0, 'few_shots'])

                if FEW_SHOTS == 10:
                    SEQ_LEN = 2048
                elif FEW_SHOTS == 25:
                    SEQ_LEN = 4096
                elif FEW_SHOTS == 50:
                    SEQ_LEN = 8192
                
                command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 ../src/fs_llm/eval-vllm.py \
                    --model_name_or_path {MODEL_NAME} \
                    --quant {QLORA_QUANT} \
                    --eval_type {EVAL_TYPE} \
                    --dataset {DATASET} \
                    --data_setting {DATA_SETTING} \
                    --few_shots {FEW_SHOTS} \
                    --task {TASK} \
                    --bf16 \
                    --max_seq_length {SEQ_LEN} \
                    --output_dir {OUTPUT_DIR} \
                    --seed {SEED} "
    
            
            process = subprocess.Popen(command, shell=True)
            process.wait()