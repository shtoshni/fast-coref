import json
from os import path


def get_data_file(data_dir, split, max_segment_len):
    jsonl_file = path.join(data_dir, "{}.{}.jsonlines".format(split, max_segment_len))
    if path.exists(jsonl_file):
        return jsonl_file
    else:
        jsonl_file = path.join(data_dir, "{}.jsonlines".format(split))
        if path.exists(jsonl_file):
            return jsonl_file

    raise ValueError(f"No relevant files at {data_dir}")


def load_dataset(
        data_dir,  singleton_file=None, max_segment_len=2048,
        num_train_docs=None, num_eval_docs=None, num_test_docs=None):
    all_splits = []
    for split in ["train", "dev", "test"]:
        jsonl_file = get_data_file(data_dir, split, max_segment_len)
        split_data = []
        with open(jsonl_file) as f:
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

    return {"train": train_data[:num_train_docs], "dev": dev_data[:num_eval_docs],
            "test": test_data[:num_test_docs]}


def load_eval_dataset(data_dir, num_test_docs, max_segment_len=2048):
    jsonl_file = get_data_file(data_dir, "test", max_segment_len)

    split_data = []
    with open(jsonl_file) as f:
        for line in f:
            split_data.append(json.loads(line.strip()))

    assert (len(split_data) >= num_test_docs)
    return {"test": split_data[:num_test_docs]}
