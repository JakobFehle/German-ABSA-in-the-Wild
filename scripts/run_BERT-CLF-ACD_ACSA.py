import subprocess
import sys
import os
import pandas as pd
import numpy as np

BATCH_SIZE = 16
RESULTS_PATH = '../results_rev/bert_clf/'
DATA_PATH = '../data/'
LEARNING_RATE = 2e-5
MODEL_NAME = 'deepset/gbert-base'

##
#Hyperparameter Validation Phase
##

EVAL_TYPE = 'dev'
SEED = 5
for DATA_SETTING in ['orig']:
    for index, (DATASET, EPOCH_RANGE) in enumerate([
                                                    ['restaurant', [20, 25, 30, 35, 40]], 
                                                    ['transport', [2, 3, 4, 5, 6]], 
                                                    ['hotel', [20, 25, 30, 35, 40]], 
                                                    ['inclusion', [20, 25, 30, 35, 40]],
                                                    ['software/v2', [20, 25, 30, 35, 40]],
    ]):
        for TASK in ['acsa', 'acd']:
            for BASE_EPOCHS in EPOCH_RANGE:
                for BATCH_SIZE in [16]:
                    
                    command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 ../src/bert_clf/classifier.py \
                    --seed {SEED} \
                    --task {TASK} \
                    --data_setting {DATA_SETTING} \
                    --dataset {DATASET} \
                    --model_name_or_path {MODEL_NAME} \
                    --learning_rate {LEARNING_RATE} \
                    --per_device_train_batch_size {BATCH_SIZE} \
                    --num_train_epochs {BASE_EPOCHS} \
                    --output_dir {RESULTS_PATH} \
                    --data_path {DATA_PATH} \
                    --eval_type {EVAL_TYPE}"
                    process = subprocess.Popen(command, shell=True)
                    process.wait()

col_names = ['task', 'dataset', 'eval_type', 'data_setting', 'learning_rate', 'batch_size', 'epoch', 'seed', 'f1-micro', 'f1-macro', 'accuracy']
runs = []

folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH)) if os.path.isdir(os.path.join(RESULTS_PATH, folder)) and folder != '.ipynb_checkpoints']

for folder_name in folder_names:
    try:
        cond_parameters = folder_name.split('_')

        if cond_parameters[0] == 'acd':
            df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_asp.tsv'), sep = '\t')
            df = df.set_index(df.columns[0])
        else:
            df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_asp_pol.tsv'), sep = '\t')
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
    for TASK in ['acsa', 'acd']:
        for DATA_SETTING in ['orig']:
            for index, DATASET in enumerate([
                'restaurant', 
                'transport', 
                'hotel', 
                'inclusion',
                'software/v2'
            ]):

                results_sub = results_all[np.logical_and.reduce([results_all['data_setting'] == f'{DATA_SETTING}-{DATA_SETTING[0]}', 
                                                            results_all['dataset'] == DATASET.replace('/', '-'),
                                                            results_all['task'] == TASK,
                                                            results_all['eval_type'] == 'dev'])].sort_values(by = ['f1-micro'], ascending = False)
                results_sub = results_sub.reset_index()
    
                print(results_sub.head(3))
                BASE_EPOCHS = int(results_sub.at[0, 'epoch'])

                command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 ../src/bert_clf/classifier.py \
                --seed {SEED} \
                --task {TASK} \
                --data_setting {DATA_SETTING} \
                --dataset {DATASET} \
                --model_name_or_path {MODEL_NAME} \
                --learning_rate {LEARNING_RATE} \
                --per_device_train_batch_size {BATCH_SIZE} \
                --num_train_epochs {BASE_EPOCHS} \
                --output_dir {RESULTS_PATH} \
                --data_path {DATA_PATH} \
                --eval_type {EVAL_TYPE}"
                process = subprocess.Popen(command, shell=True)
                process.wait()