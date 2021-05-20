import json
from os import path


def load_dataset(data_dir, max_segment_len=2048, dataset='litbank',
                 num_train_docs=None, num_eval_docs=None, singleton_file=None, **kwargs):
    all_splits = []
    for split in ["train", "dev", "test"]:
        jsonl_file = path.join(data_dir, "{}.{}.jsonlines".format(split, max_segment_len))
        if not path.exists(jsonl_file):
            jsonl_file = path.join(data_dir, "{}.jsonlines".format(split))
        split_data = []
        with open(jsonl_file) as f:
            for line in f:
                split_data.append(json.loads(line.strip()))
        all_splits.append(split_data)

    train_data, dev_data, test_data = all_splits

    if dataset == 'ontonotes' and singleton_file is not None and path.exists(singleton_file):
        num_singletons = 0
        with open(singleton_file) as f:
            singleton_data = json.loads(f.read())

        for instance in train_data:
            doc_key = instance['doc_key']
            if doc_key in singleton_data:
                num_singletons += len(singleton_data[doc_key])
                instance['clusters'].extend(singleton_data[doc_key])

        print("Added %d singletons" % num_singletons)

    return {"train": train_data[:num_train_docs], "dev": dev_data[:num_eval_docs], "test": test_data[:num_eval_docs]}


def load_eval_dataset(data_dir, dataset='quizbowl', max_segment_len=2048, num_eval_docs=None, split="test"):
    jsonl_file = path.join(data_dir, "{}.{}.jsonlines".format(split, max_segment_len))
    if not path.exists(jsonl_file):
        jsonl_file = path.join(data_dir, "{}.jsonlines".format(split))
    split_data = []
    with open(jsonl_file) as f:
        for line in f:
            split_data.append(json.loads(line.strip()))

    if split == 'test':
        if dataset == 'quizbowl':
            assert(len(split_data) == 400)
        elif dataset == 'ontonotes':
            assert (len(split_data) == 348)
        elif dataset == 'preco':
            assert (len(split_data) == 500)
        elif dataset == 'wikicoref':
            assert (len(split_data) == 30)
        elif dataset == 'litbank':
            assert (len(split_data) == 10)
        elif dataset == 'wsc':
            assert (len(split_data) == 271)
        elif dataset == 'gap':
            assert (len(split_data) == 2000)
        elif dataset == 'character_identification':
            assert (len(split_data) == 192)
        else:
            pass

    return {split: split_data[:num_eval_docs]}
