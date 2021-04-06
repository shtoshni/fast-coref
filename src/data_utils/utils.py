import json
from os import path


def load_data(data_dir, max_segment_len, dataset='litbank', singleton_file=None,
              num_train_docs=None, num_eval_docs=None):
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

    return {"train": train_data[:num_train_docs], "dev": dev_data[:num_eval_docs], "test": test_data[:num_eval_docs]}


