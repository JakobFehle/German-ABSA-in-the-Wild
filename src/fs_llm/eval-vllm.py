import pandas as pd
import bitsandbytes as bnb
import torch
import numpy as np
import os
import time
import json
import sys
import re

utils = os.path.abspath('../src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from helpers_llm import *
from transformers import set_seed
from preprocessing import loadDataset, splitForEvalSetting
from evaluation import (
    createResults, convertLabels
)
from vllm import LLM, SamplingParams
from ast import literal_eval
from argparse import ArgumentParser

HF_TOKEN = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REGEX_ASPECTS_ACSD = r"\(\'([^\']+)\',\s*([^,]+),\s*([^\']+)\)"

def createSamplingParams(config):
    STOP_WORDS = ['Text:', '\n\n', 'Sentiment-Elemente:']
    
    return SamplingParams(
        temperature = config.temperature, 
        stop = STOP_WORDS,
        max_tokens = config.max_new_tokens,
        top_k = config.top_k,
        top_p = config.top_p,
        skip_special_tokens = True
    )

def safe_recursive_pattern(depth, max_depth):
    quoted_content = r'"(?:[^"\\]|\\.)*"'
    
    if depth >= max_depth:
        return rf'(?:{quoted_content}|[^()])*'
    return rf'\((?:{quoted_content}|[^()]|{safe_recursive_pattern(depth + 1, max_depth)})*\)'

def extractAspects(output, task):
    result = []
    if task == 'tasd':
        max_depth = 5
        pattern_targets = re.compile(safe_recursive_pattern(0, max_depth))
        pairs = pattern_targets.findall(output)

        for pair in pairs:
            try: 
                match = literal_eval(pair)
                result.append([match[1], match[2], match[0]])
            except:
                pass

        return result
    
    elif task == 'acd':
        REGEX_ASPECTS_ACD = r"\[([^\]]+)\]"
        pattern_asp = re.compile(REGEX_ASPECTS_ACD)
        matches = pattern_asp.findall(output)

        for match in matches:
            aspects = [s.strip().strip("'\"") for s in match.split(',')]
            result.extend(aspects)
        return result

    elif task == 'acsa':
        REGEX_ASPECTS_ACSA = r"\(([^,]+),\s*([^,]+)\)"
        pattern_pairs = re.compile(REGEX_ASPECTS_ACSA)
        
        pairs = pattern_pairs.findall(output)

        for pair in pairs:
            result.append([pair[0], pair[1]])
        return result
        
    else:
        return []


def createModel(config):
    """
    Loads a pretrained model and tokenizer with LoRA adaptation.

    Args:
        config (Config): Model and LoRA configuration parameters.

    Returns:
        Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]: LoRA-adapted model and tokenizer.
    """

    model = LLM(
        model=config.model_name_or_path, 
        tokenizer=config.model_name_or_path,
        dtype='bfloat16',
        max_model_len=config.max_seq_length,
        tensor_parallel_size=torch.cuda.device_count(),
        seed = config.seed,
        gpu_memory_utilization = 0.9
    )

    return model

def savePredictions(result, preds, golds, config, eval_stats, eval_type):
    """
    Saves predictions, metrics, and configuration details to disk.

    Args:
        result (List[Dict]): Computed evaluation metrics.
        preds (List): Model predictions.
        golds (List): Ground truth labels.
        config (Config): Configuration object.
        training_args (TrainingArguments): Huggingface training args.
        eval_type (str): Evaluation dataset suffix (e.g., 'orig', 'dia').
    """

    output_path = f"{config.output_dir}/{config.task}_{config.dataset.replace('/', '-')}_{config.eval_type}_{config.data_setting}-{eval_type[0]}_{config.few_shots}_{config.seed}/"
    os.makedirs(output_path, exist_ok=True)

    # Save individual metric outputs
    if config.task == 'acd':
        TASKS = ["asp"]
    elif config.task == 'acsa':
        TASKS = ["asp", "asp_pol", "pairs", "pol"]
    elif config.task == 'tasd':
        TASKS = ["asp", "asp_pol", "pairs", "pol", "phrases"]

    for idx, name in enumerate(TASKS):
        pd.DataFrame.from_dict(result[idx]).transpose().to_csv(f"{output_path}metrics_{name}.tsv", sep="\t")

    try:
        # Save predictions and gold labels for further analysis
        matched_samples = [
            {"predictions": pred, "gold_labels": gold}
            for pred, gold in zip(preds, golds)
        ]
        with open(os.path.join(output_path, 'predictions.json'), "w", encoding="utf-8") as f:
            json.dump({"test": matched_samples}, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save predictions. {e}")

    # Save config used for this run
    with open(os.path.join(output_path, 'config.json'), "w", encoding="utf-8") as f:
        json.dump(eval_stats, f, indent=4, ensure_ascii=False)


def evaluate(config, model, prompts_test, ground_truth, label_space):
    """
    Runs evaluation by generating predictions and computing metrics.

    Args:
        config (Config): Configuration with evaluation parameters.
        model (torch.nn.Module): Trained model.
        prompts_test (Dataset): Dataset containing test prompts.
        gold_labels (List): Ground truth labels.
        label_space (List): Set of valid label classes.
        sampling_params (SamplingParams): Generation parameters.

    Returns:
        Tuple[List[Dict], List, List]: Evaluation results, gold labels, and model outputs.
    """
    sampling_params = createSamplingParams(config)

    try:
        if config.task == 'tasd':
            ground_truth = [[(asp, pol, phr) for asp, pol, phr in labels] for labels in ground_truth]
        elif config.task == 'acsa':
            ground_truth = [[(asp, pol) for asp, pol, phr in labels] for labels in ground_truth]
        elif config.task == 'acd':
            ground_truth = [[asp for asp, pol, phr in labels] for labels in ground_truth]
    except:
        if config.task == 'acsa':
            ground_truth = [[(asp, pol) for asp, pol in labels] for labels in ground_truth]
        elif config.task == 'acd':
            ground_truth = [[asp for asp, pol in labels] for labels in ground_truth]
    
    print(len(prompts_test))
        
    time_start = time.time()

    model_outputs = model.generate(prompts_test, sampling_params)
    
    time_end = time.time()
    outputs = [out.outputs[0].text for out in model_outputs]
    print(model_outputs[0].outputs[0])
    predictions = [extractAspects(out.outputs[0].text, config.task) for out in model_outputs]
    print(predictions[0])
    # Convert predictions and gold labels into evaluation format
    gold_labels, _ = convertLabels(ground_truth, config.task, label_space)
    pred_labels, false_predictions = convertLabels(predictions, config.task, label_space)
    print(pred_labels[0])
    print(gold_labels[0])
    print(label_space)
    # Compute evaluation metrics
    results = createResults(pred_labels, gold_labels, label_space, config.task)
    return results, ground_truth, [{"output": out, "predictions": pred} for out, pred in zip(outputs, predictions)], time_end - time_start

def createPrompts(task, labels, train, test, few_shots, eos_token):
    PROMPT_TEMPLATE = globals()[f"PROMPT_{task.upper()}"]

    prompts_train = []
    prompts_test = []
    ground_truth = []

    print(f"Creating prompts with {few_shots} few shots", )
    
    few_shot_text = ''
    if few_shots > 0:
        
        few_shots = train[['text', 'labels']].sample(n=few_shots, random_state=42)
        
        for index, row in few_shots.iterrows():
            try:
                if task == 'tasd':
                    new_labels = [(phr, asp, pol) for asp, pol, phr in row['labels']]
                elif task == 'acsa':
                    new_labels = [(asp, pol) for asp, pol, phr in row['labels']]
                else:
                    new_labels = [asp for asp, pol, phr in row['labels']]
                    text = row['text'][:500] # Limit Example-Text to 500 Characters
            except:
                if task == 'acsa':
                    new_labels = [(asp, pol) for asp, pol in row['labels']]
                else:
                    new_labels = [asp for asp, pol in row['labels']]
            text = row['text'][:500] # Limit Example-Text to 500 Characters
            few_shot_text += f"Text: {text}\nSentiment-Elemente: {str(new_labels)}\n\n"

    for index, row in train.iterrows():
        try:
            if task == 'tasd':
                new_labels = [(phr, asp, pol) for asp, pol, phr in row['labels']]
            elif task == 'acsa':
                new_labels = [(asp, pol) for asp, pol, phr in row['labels']]
            else:
                new_labels = [asp for asp, pol, phr in row['labels']]
        except:
            if task == 'acsa':
                new_labels = [(asp, pol) for asp, pol in row['labels']]
            else:
                new_labels = [asp for asp, pol in row['labels']]

        prompt = PROMPT_TEMPLATE.format(categories = labels, examples=few_shot_text)
        prompt_text = row['text'][:500]
        prompt += f"Text: {prompt_text}\nSentiment-Elemente: {str(new_labels)}" + eos_token
        
        prompts_train.append(prompt)

    for index, row in test.iterrows():
        prompt = PROMPT_TEMPLATE.format(categories = labels, examples=few_shot_text)
        prompt_text = row['text'][:500]
        prompt += f"Text: {prompt_text}\nSentiment-Elemente: "
    
        prompts_test.append(prompt)
        ground_truth.append(row['labels'])

    return prompts_train, prompts_test, ground_truth
    
def main():
    """
    Main entry point: loads configuration, prepares data, trains model, evaluates and saves results.
    """
    config = Config()
    print(', '.join("%s: %s" % item for item in vars(config).items()))

    set_seed(config.seed)
    
    eval_stats = {
        "model_name": config.model_name_or_path,
        "task": config.task,
        "data_setting": config.data_setting,
        "dataset": config.dataset,
        "eval_type": config.eval_type,
        "seed": config.seed,
        "temperature": config.temperature, 
        "max_tokens": config.max_new_tokens
    }
    
    # Load model and tokenizer
    model = createModel(config)

    # Prepare datasets and prompts
    df_train, df_test, label_space = splitForEvalSetting(loadDataset(config.data_path, config.dataset), config.eval_type)
    categories = list(set([label.split(':')[0] for label in label_space]))
    prompts_train, prompts_test, ground_truth_labels = createPrompts(config.task, categories, df_train, df_test, config.few_shots, eos_token='')

    print(prompts_test[0])
    # Generate evaluation samples
    results, all_labels, all_preds, eval_time = evaluate(config, model, prompts_test, ground_truth_labels, label_space)
    print(results)
    eval_stats.update({"eval_time": eval_time})
    savePredictions(results, all_preds, all_labels, config, eval_stats, 'orig')

    # Optionally evaluate on dialectal test split (if applicable)
    if 'test' in config.eval_type and config.dataset == 'transport':
        _, df_test_dia, _ = splitForEvalSetting(loadDataset(config.data_path, config.dataset, 'dia'), config.eval_type)
        _, prompts_test_dia, ground_truth_labels_dia = createPrompts(config.task, categories, df_train, df_test_dia, config.few_shots, eos_token='')
        results, all_labels, all_preds, eval_time = evaluate(config, model, prompts_test_dia, ground_truth_labels_dia, label_space)
        eval_stats.update({"eval_time": eval_time})
        savePredictions(results, all_preds, all_labels, config, eval_stats, 'dia')

class Config(object):
    def __init__(self):

        # General Params
        self.task = None
        self.output_dir = '../results/'
        self.data_path = '../data/'
        
        # Dataset Params
        self.dataset = None
        self.eval_type = None
        self.data_setting = None
        self.few_shots = 0
        
        # Inference Params
        self.quant = 4
        self.max_new_tokens = 200
        self.temperature = 0
        self.top_k = -1
        self.top_p = 1
        
        # Model Params
        self.model_name_or_path = "meta-llama/Meta-Llama-3-8B"
        self.seed = 42
        self.bf16 = True
        self.max_seq_length = 2048
        
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.update_config_with_args()

    def update_config_with_args(self):
        for key, value in self.args.items():
            if value is not None:
                setattr(self, key, value)
                
    def setup_parser(self):

        parser = ArgumentParser()

        # Model-related arguments
        parser.add_argument('--model_name_or_path', type=str, help='Base model name for training or inference')
        parser.add_argument('--seed', type=int, help='Seed to ensure reproducability.')
        parser.add_argument('--quant', type=int, help="How many bits to use for quantization.")
        parser.add_argument('--bf16', action='store_true', help="Compute dtype of the model (uses bf16 if set).")
        parser.add_argument('--output_dir', type=str, help='Relative path to output directory.')
        # Dataset-related arguments
        parser.add_argument('--dataset', type=str, required=True, help="Which dataset to use: ['hotel', 'rest' or 'germeval']")
        parser.add_argument('--max_seq_length', type=int, help="Maximum context length during training and inference.")
        parser.add_argument('--task', type=str, default="acsa", help="Which ABSA Task the model was trained on. ['acd', 'acsa', 'acsd']")
        parser.add_argument('--eval_type', type=str, default="dev", help="Which dataset split used for evaluation")
        parser.add_argument('--few_shots', type=int, default=0)
        parser.add_argument('--data_setting', type=str, default='orig')
        parser.add_argument('--data_path', type=str, default='../data/')

        # Inference-related arguments
        parser.add_argument('--max_new_tokens', type=int, help="Maximum sequence length for new tokens during inference.")
        parser.add_argument('--temperature', type=float, help="Temperature for sampling.")
        parser.add_argument('--top_k', type=float, help="Top-k sampling parameter.")
        parser.add_argument('--top_p', type=float, help="Top-p sampling parameter.")
        
        return parser

if __name__ == "__main__":
    main()