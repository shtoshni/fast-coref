# Constants used in the project

CANONICAL_CLUSTER_THRESHOLD = \
    {'litbank': 1, 'preco': 1, 'quizbowl': 1, 'wikicoref': 2, 'ontonotes': 2,
     'cd2cr': 1, 'wsc': 1, 'character_identification': 1}

SPEAKER_START = '[SPEAKER_START]'
SPEAKER_END = '[SPEAKER_END]'


MODEL_TO_MAX_LEN = {
    'longformer': 4096,
    'spanbert': 512,
}

MODEL_TO_MODEL_STR = {
    'longformer': 'allenai/longformer-large-4096',
    'spanbert': 'bert-base-cased',
}


