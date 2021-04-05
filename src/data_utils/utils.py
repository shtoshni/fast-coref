import json
from os import path
from transformers import LongformerTokenizerFast
from data_utils.coref_dataset import CorefDataset
import torch


def load_data(data_dir, max_segment_len, dataset='litbank', singleton_file=None,
              num_train_docs=None, num_eval_docs=None, max_training_segments=None,
              num_workers=0, training=True):
    all_splits = []
    for split in ["train", "dev", "test"]:
        jsonl_file = path.join(data_dir, "{}.{}.jsonlines".format(split, max_segment_len))
        with open(jsonl_file) as f:
            split_data = []
            for line in f:
                split_data.append(json.loads(line.strip()))
        all_splits.append(split_data)

    train_data, dev_data, test_data = all_splits

    if singleton_file is not None and path.exists(singleton_file):
        num_singletons = 0
        with open(singleton_file) as f:
            singleton_data = json.loads(f.read())

        for instance in train_data:
            doc_key = instance['doc_key']
            if doc_key in singleton_data:
                num_singletons += len(singleton_data[doc_key])
                instance['clusters'].extend(singleton_data[doc_key])

        print("Added %d singletons" % num_singletons)

    if dataset == 'litbank':
        assert(len(train_data) == 80)
        assert(len(dev_data) == 10)
        assert(len(test_data) == 10)
    elif dataset == 'ontonotes':
        assert (len(train_data) == 2802)
        assert (len(dev_data) == 343)
        assert (len(test_data) == 348)

    tokenizer = LongformerTokenizerFast.from_pretrained(f'allenai/longformer-large-4096', add_prefix_space=False)

    if training:
        train_dataset = CorefDataset(train_data[:num_train_docs], tokenizer,
                                     max_training_segments=max_training_segments)
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset, num_workers=num_workers, pin_memory=True,
                batch_size=None, shuffle=True,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            CorefDataset(train_data[:num_train_docs], tokenizer), num_workers=num_workers,
            batch_sampler=None, batch_size=None, shuffle=False, pin_memory=True,
        )

    val_dataloader = torch.utils.data.DataLoader(
        CorefDataset(dev_data[:num_eval_docs], tokenizer), num_workers=num_workers,
        batch_sampler=None, batch_size=None, shuffle=False, pin_memory=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        CorefDataset(test_data[:num_eval_docs], tokenizer), num_workers=0, pin_memory=True,
        batch_sampler=None, batch_size=None, shuffle=False,
    )

    return {"train": train_dataloader, "dev": val_dataloader, "test": test_dataloader}
