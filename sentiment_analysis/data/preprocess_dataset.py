import argparse
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dataset")
    parser.add_argument("output_path")

    args = parser.parse_args()
    source_dataset = args.source_dataset
    output_path = args.output_path

    dataset_dict = datasets.load_dataset(
        source_dataset,
        split={
            "train": datasets.ReadInstruction("train", from_=0, to=65, unit="%"),
            "validation": datasets.ReadInstruction("train", from_=65, to=80, unit="%"),
            "test": datasets.ReadInstruction("train", from_=80, to=100, unit="%"),
        },  # type: ignore
    )

    assert isinstance(dataset_dict, datasets.DatasetDict)
    dataset_dict.save_to_disk(output_path)
    print(dataset_dict)
