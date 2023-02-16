from typing import Dict, List, Literal
import argparse
import datasets

from collections import Counter
import os

MappedDatasetKeys = Literal["sentence", "label"]
MappedDatasetBatch = Dict[MappedDatasetKeys, List]
# LABEL_MAP = {0: 0, 2: 1, 1: 0, 4: 2, 3: 2}
# LABEL_MAP = {0: 0, 2: 1, 4: 2}
LABEL_MAP = {0: 0, 2: 1}


def convert_batch(batch: Dict[str, List]) -> MappedDatasetBatch:
    """
    Map/filter labels.
    """
    output: MappedDatasetBatch = {"sentence": [], "label": []}
    texts = batch["text"]
    labels = batch["label"]

    for text, label in zip(texts, labels):
        mapped_label = LABEL_MAP.get(label)
        if mapped_label is not None:
            output["sentence"].append(text)
            output["label"].append(mapped_label)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset_path_or_name")
    parser.add_argument("output_dataset_path")
    args = parser.parse_args()

    input_dataset_path_or_name = args.input_dataset_path_or_name
    output_dataset_path = args.output_dataset_path

    if os.path.isdir(input_dataset_path_or_name):
        dataset = datasets.load_from_disk(input_dataset_path_or_name)
    else:
        dataset = datasets.load_dataset(input_dataset_path_or_name)

    assert isinstance(dataset, datasets.DatasetDict)
    print(dataset)

    mapped_dataset = dataset.map(
        convert_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(mapped_dataset)
    print(Counter(dataset["train"]["label"]))
    print(Counter(mapped_dataset["train"]["label"]))
    mapped_dataset.save_to_disk(output_dataset_path)
