import unittest
import os

import datasets
import jax
from transformers import AutoTokenizer

from ..data.rm_dataloader import get_dataloader

DATASET_DICT_PATH = os.environ.get("TEST_DATASET_DICT_PATH", "data/processed/semeval_3")
DATASET_DICT_SPLIT = os.environ.get("TEST_DATASET_DICT_SPLIT", "validation")
TOKENIZER_PATH = os.environ.get("TEST_TOKENIZER", "roberta-base")
EXAMPLE_BATCH_SIZE = int(os.environ.get("TEST_EXAMPLE_BATCH_SIZE", "16"))
EXAMPLE_BLOCK_SIZE = int(os.environ.get("TEST_EXAMPLE_BLOCK_SIZE", "256"))


def _print_tree_shape(tree):
    shape = jax.tree_util.tree_map(jax.numpy.shape, tree)
    print(shape)


class DataloaderTestCases(unittest.TestCase):
    def setUp(self):
        self.test_dataset: datasets.Dataset = datasets.load_from_disk(
            DATASET_DICT_PATH
        )[
            DATASET_DICT_SPLIT
        ]  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    def test_dataloader_without_shuffling(self):
        num_batches, init_dataloader = get_dataloader(
            self.test_dataset,
            self.tokenizer,
            EXAMPLE_BATCH_SIZE,
            EXAMPLE_BLOCK_SIZE,
            1.1,
            None,
        )

        # print(num_batches)
        dataloader = init_dataloader()
        batch = next(dataloader)
        _print_tree_shape(batch)

        # print(self.tokenizer.decode(batch.chosen.input_ids[0, :]))
        # print("===")
        # print(self.tokenizer.decode(batch.rejected.input_ids[0, :]))

    def test_dataloader_with_shuffling(self):
        example_prng_key = jax.random.PRNGKey(0)

        num_batches, init_dataloader = get_dataloader(
            self.test_dataset,
            self.tokenizer,
            EXAMPLE_BATCH_SIZE,
            EXAMPLE_BLOCK_SIZE,
            1.1,
            example_prng_key,
        )

        # print(num_batches)
        dataloader = init_dataloader()
        batch = next(dataloader)
        _print_tree_shape(batch)

        # print(self.tokenizer.decode(batch.chosen.input_ids[0, :]))
        # print(batch.chosen.input_ids[0, :])
        # print(batch.rejected.input_ids[0, :])
        # print("===")
        # print(self.tokenizer.decode(batch.rejected.input_ids[0, :]))
