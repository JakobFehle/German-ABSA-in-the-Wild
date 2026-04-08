import subprocess
import sys
import os
import pandas as pd
import numpy as np

LEARNING_RATE = 1e-4
DATA_PATH = '../data/'
OUTPUT_PATH = '../results/mvp/'
TASK = 'tasd'
BATCH_SIZE = 16
GRADIENT_STEPS = 1
MODEL = 't5-base'

EVAL_TYPE = 'dev'
for SEED in [5]:
    for DATA_SETTING in ['orig']:
        for DATASET, EPOCH_RANGE in [
                ['inclusion', [10, 15, 20, 25]],
                ['restaurant', [10, 15, 20, 25]], 
                ['software/v2', [10, 15, 20, 25]],
                ['transport', [2, 3, 4, 5]],
                                        
                                    ]:
            for BASE_EPOCHS in EPOCH_RANGE:
                command = [
                    sys.executable,  # The Python interpreter
                    "../src/mvp/src/classifier.py",  # The script to run
                    "--data_path", DATA_PATH,
                    "--model_name_or_path", MODEL,
                    "--dataset", DATASET,
                    "--eval_type", EVAL_TYPE,
                    "--data_setting", DATA_SETTING,
                    "--output_dir", OUTPUT_PATH,
                    "--num_train_epochs", str(BASE_EPOCHS),
                    "--save_top_k", "0",
                    "--task", TASK,
                    "--top_k", "5",
                    "--ctrl_token", "post",
                    "--multi_path",
                    "--num_path", "5",
                    "--seed", str(SEED),
                    "--train_batch_size", str(BATCH_SIZE),
                    "--gradient_accumulation_steps", str(GRADIENT_STEPS),
                    "--learning_rate", str(LEARNING_RATE),
                    "--sort_label",
                    "--data_ratio", "1.0",
                    "--check_val_every_n_epoch", str(BASE_EPOCHS+1),
                    "--agg_strategy", "vote",
                    "--eval_batch_size", "16",
                    "--constrained_decode",
                    # "--lowercase"
                ]
                
                # Add the environment variable as a prefix
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
                env["TOKENIZERS_PARALLELISM"] = 'false'
                
                # Run the subprocess
                process = subprocess.Popen(command, env=env)
                process.wait()


RESULTS_PATH = '../results/mvp/'
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
for SEED in [5, 10, 15, 20, 25]:        
    for DATA_SETTING in ['orig']:
        for index, DATASET in enumerate([
            'restaurant', 
            'software/v2',
            'transport', 
            'inclusion'
        ]):

            results_sub = results_all[np.logical_and.reduce([results_all['data_setting'] == f'{DATA_SETTING}-{DATA_SETTING[0]}', 
                                                        results_all['dataset'] == DATASET.replace('/', '-'),
                                                        results_all['task'] == TASK,
                                                        results_all['eval_type'] == 'dev'])].sort_values(by = ['f1-micro'], ascending = False)
            results_sub = results_sub.reset_index()

            print(results_sub.head(3))
            BASE_EPOCHS = int(results_sub.at[0, 'epoch'])

            command = [
                sys.executable,  # The Python interpreter
                "../src/mvp/src/classifier.py",  # The script to run
                "--data_path", DATA_PATH,
                "--model_name_or_path", MODEL,
                "--dataset", DATASET,
                "--eval_type", EVAL_TYPE,
                "--data_setting", DATA_SETTING,
                "--output_dir", OUTPUT_PATH,
                "--num_train_epochs", str(BASE_EPOCHS),
                "--save_top_k", "0",
                "--task", TASK,
                "--top_k", "5",
                "--ctrl_token", "post",
                "--multi_path",
                "--num_path", "5",
                "--seed", str(SEED),
                "--train_batch_size", str(BATCH_SIZE),
                "--gradient_accumulation_steps", str(GRADIENT_STEPS),
                "--learning_rate", str(LEARNING_RATE),
                "--sort_label",
                "--data_ratio", "1.0",
                "--check_val_every_n_epoch", str(BASE_EPOCHS+1),
                "--agg_strategy", "vote",
                "--eval_batch_size", "16",
                "--constrained_decode",
                # "--lowercase"
            ]
            
            # Add the environment variable as a prefix
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
            env["TOKENIZERS_PARALLELISM"] = 'false'
            
            # Run the subprocess
            process = subprocess.Popen(command, env=env)
            process.wait()
