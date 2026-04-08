import pandas as pd

from itertools import product
from ast import literal_eval
from sklearn.model_selection import KFold

LABEL_SPACES = [
    ###
    #   Restaurant Reviews
    ###
    ['food', 'service', 'general impression', 'ambience', 'price'],

    ###
    #   Hotel Reviews
    ###
    ['hotel', 'rooms', 'location', 'food', 'service'],

    ###
    #   DB Reviews
    ###
    ['general', 'miscellaneous', 'tickets', 'safety', 'atmosphere',
     'ride', 'information', 'capacity', 'app', 'service',
     'connectivity', 'child friendliness', 'comfort', 'design', 'toilets',
     'accessibility', 'luggage', 'food', 'qr code', 'image'],

    ###
    #   Reviews about Inclusion in Transport
    ###
    ['space', 'lift', 'lighting', 'accidents', 'barrier general',
     'escalator', 'info display', 'info', 'info acoustic', 'guiding routes', 
     'barrier others', 'ground level access', 'construction site', 'ramp', 'demonstration',
     'security', 'acoustic signal'],


    ###
    #   B2B Software Reviews
    ###
    ['ease of use', 'interface design', 'general experience', 'technical performance', 'pricing',
     'functional scope', 'customer support']
    
]

DATASETS = ['restaurant', 'hotel', 'transport', 'inclusion', 'software']

POLARITY_MAPPING_POL_TO_TERM = {"negative": "schlecht", "neutral": "okay", "positive": "gut"}

POLARITY_MAPPING_TERM_TO_POL = {"schlecht": "negative", "okay": "neutral", "gut": "positive"}

OUTPUT_KEYS = ['per_device_train_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm', 'num_train_epochs', 'lr_scheduler_type', 'warmup_steps', 'seed', 'bf16', 'fp16', 'group_by_length', '_n_gpu', 'generation_max_length']

TEXT_TEMPLATE = "{ac_text} ist {polarity_text} weil {aspect_term_text} {polarity_text} ist"

TEXT_PATTERN = r"(.*) ist (.*) weil (.*) (.*) ist"

IT_TOKEN = "es"

REGEX_ASPECTS_ACSD = r"\(([^,]+),[^,]+,\s*\"[^\"]*\"\)"
REGEX_LABELS_ACSD = r"\([^,]+,\s*([^,]+)\s*,\s*\"[^\"]*\"\s*\)"
REGEX_PHRASES_ACSD = r"\([^,]+,\s*[^,]+\s*,\s*\"([^\"]*)\"\s*\)"
REGEX_ASPECTS = r"\(([^,]+),[^)]+\)"
REGEX_LABELS = r"\([^,]+,\s*([^)]+)\)"

def raise_err(ex):
    raise ex

def splitForEvalSetting(dataset, eval_type):
    """Handles dataset splitting and cross-validation settings."""
    (train, dev), test, label_space = dataset
        
    if 'dev' in eval_type and dev is not None:
        test = dev
    elif 'dev' in eval_type:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = list(kf.split(train, None))[0]
        train, test = train.iloc[train_idx], train.iloc[val_idx]
        print(f"Creating dev split; using split {0} with random_state 42")

    print(f"{eval_type} mode")
    print(f"Using Train set size: {len(train)}, Dev/Test set size: {len(test)}")
    return train, test, label_space

def loadDataset(data_path, dataset, test = 'orig'):

    for index, dataset_name in enumerate(DATASETS):
        if dataset_name in dataset:
            label_space = LABEL_SPACES[index]
            label_space = [f'{category}:{polarity}' for category in label_space for polarity in ['positive', 'neutral', 'negative']]

    df_dev = None
    setting = '' if test == 'orig' else '_dia'
        
    path_train = f'{data_path}/{dataset}/train.json'
    path_eval = f'{data_path}/{dataset}/test{setting}.json'
    
    df_train = pd.read_json(path_train, orient="records", lines=True).set_index('id')
    df_eval = pd.read_json(path_eval, orient="records", lines=True).set_index('id')
    
    
    try:
        path_dev = f'{data_path}/{dataset}/dev.json'
        df_dev = pd.read_json(path_dev, orient="records", lines=True).set_index('id')
    except:
        pass
        
    print(f'Loading dataset ...')
    print(f'Dataset: {dataset}')
    print(f'Setting: {setting}')
    print(f'Train Length: ', len(df_train))
    if df_dev is not None:
        print(f'Dev Length: ', len(df_dev))
    print(f'Test Length: ', len(df_eval))
        
    return [df_train, df_dev], df_eval, label_space