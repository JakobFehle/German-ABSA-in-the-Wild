import subprocess
import os
import pandas as pd
import numpy as np
import sys

RESULTS_PATH = '../results/paraphrase/'
TASK = 'tasd'
BATCH_SIZE = 16
GRADIENT_STEPS = 1
LEARNING_RATE = 3e-4
DATA_PATH = '../data/'
MODEL_NAME = 't5-base'

###
# Hyperparameter Validation Phase
###

SEED = 5
EVAL_TYPE = 'dev'
    
for SEED in [5]:
    for DATA_SETTING in ['orig']:
        for index, (DATASET, EPOCH_RANGE) in enumerate([
                                                ['restaurant', [15, 20, 25, 30]], 
                                                ['transport', [2, 3, 4, 5, 6]], 
                                                ['hotel', [15, 20, 25, 30]], 
                                                ['inclusion', [15, 20, 25, 30]],
                                                ['software/v2', [15, 20, 25, 30]],
        ]):
            for BASE_EPOCHS in EPOCH_RANGE:
            
                command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 ../src/paraphrase/classifier.py \
                --task {TASK} \
                --data_path {DATA_PATH} \
                --data_setting {DATA_SETTING} \
                --dataset {DATASET} \
                --learning_rate {LEARNING_RATE} \
                --per_device_train_batch_size {BATCH_SIZE} \
                --num_train_epochs {BASE_EPOCHS} \
                --model_name_or_path {MODEL_NAME} \
                --output_dir {RESULTS_PATH} \
                --gradient_accumulation_steps {GRADIENT_STEPS} \
                --seed {SEED} \
                --eval_type {EVAL_TYPE}"
                process = subprocess.Popen(command, shell=True)
                process.wait()


RESULTS_PATH = '../results/paraphrase/'
col_names = ['task', 'dataset', 'eval_type', 'data_setting', 'learning_rate', 'batch_size', 'epoch', 'seed', 'f1-micro', 'f1-macro', 'accuracy']
runs = []

folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH)) if os.path.isdir(os.path.join(RESULTS_PATH, folder)) and folder != '.ipynb_checkpoints']

for folder_name in folder_names:
    try:
        cond_parameters = folder_name.split('_')

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

EVAL_TYPE = 'test'

for SEED in [5,10,15,20,25]:        
    for DATA_SETTING in ['orig']:
        for index, DATASET in enumerate([
            'restaurant', 
            'transport', 
            'hotel', 
            'inclusion'
            'software/v2'
        ]):

            results_sub = results_all[np.logical_and.reduce([results_all['data_setting'] == f'{DATA_SETTING}-{DATA_SETTING[0]}', 
                                                        results_all['dataset'] == DATASET.replace('/', '-'),
                                                        results_all['task'] == TASK,
                                                        results_all['eval_type'] == 'dev'])].sort_values(by = ['f1-micro'], ascending = False)
            results_sub = results_sub.reset_index()

            print(results_sub.head(3))
            BASE_EPOCHS = int(results_sub.at[0, 'epoch'])

            command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 ../src/paraphrase/classifier.py \
                    --task {TASK} \
                    --data_path {DATA_PATH} \
                    --data_setting {DATA_SETTING} \
                    --dataset {DATASET} \
                    --learning_rate {LEARNING_RATE} \
                    --per_device_train_batch_size {BATCH_SIZE} \
                    --num_train_epochs {BASE_EPOCHS} \
                    --model_name_or_path {MODEL_NAME} \
                    --output_dir {RESULTS_PATH} \
                    --gradient_accumulation_steps {GRADIENT_STEPS} \
                    --seed {SEED} \
                    --eval_type {EVAL_TYPE}"
            process = subprocess.Popen(command, shell=True)
            process.wait()