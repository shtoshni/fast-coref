SPEAKER_START = '[SPEAKER_START]'
SPEAKER_END = '[SPEAKER_END]'

MODEL_TO_MAX_LEN = {
    'longformer': 4096,
    'spanbert': 512,
    'bigbird': 4096,
    'bert-dutch': 512,
}

MODEL_TO_MODEL_STR = {
    'longformer': 'allenai/longformer-large-4096',
    'spanbert': 'bert-base-cased',
    'bigbird': 'flax-community/pino-bigbird-roberta-base',
    'bert-dutch': 'GroNLP/bert-base-dutch-cased'
}