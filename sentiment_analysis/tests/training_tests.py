import unittest
import os

import datasets
from transformers import FlaxRobertaForSequenceClassification, AutoTokenizer

import jax
import optax
from flax.training.train_state import TrainState

from ..data.rm_dataloader import get_dataloader
from ..models.train_classifier import jit_train_step, loss_accuracy_fn, grad_fn

HF_MODEL_NAME = os.environ.get("TEST_HF_MODEL", "roberta-base")
HF_TOKENIZER_NAME = os.environ.get("TEST_HF_TOKENIZER", HF_MODEL_NAME)
BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "8"))
BLOCK_SIZE = int(os.environ.get("TEST_BLOCK_SIZE", "256"))
NUM_TRAIN_STEPS = int(os.environ.get("TEST_NUM_TRAIN_STEPS", "12"))
LEARNING_RATE = float(os.environ.get("TEST_LERANING_RATE", "0.0001"))
DATASET_PATH = os.environ.get("TEST_DATASET_DICT_PATH", "data/processed/semeval_3")


class RMTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_dict = datasets.load_from_disk(DATASET_PATH)
        assert isinstance(dataset_dict, datasets.DatasetDict)
        dataset = dataset_dict["train"]

        tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
        _, cls.init_train_dataloader = get_dataloader(
            dataset,
            tokenizer,
            BATCH_SIZE,
            BLOCK_SIZE,
            0.1,
            jax.random.PRNGKey(0),
        )

        (
            cls.model,
            initial_params,
        ) = FlaxRobertaForSequenceClassification.from_pretrained(
            HF_MODEL_NAME, _do_init=False, num_labels=3
        )  # type: ignore
        assert isinstance(cls.model, FlaxRobertaForSequenceClassification)

        initial_params = jax.device_put(initial_params, jax.devices()[0])
        cls.params = cls.model.init_weights(
            jax.random.PRNGKey(0),
            (1, 1),
            initial_params,  # type: ignore
        )

    def test_loss_accuracy_fn(self):
        dataloader = RMTrainingTests.init_train_dataloader()
        example_batch = next(dataloader)

        loss, accuracy = loss_accuracy_fn(
            RMTrainingTests.model.__call__,
            example_batch,
            RMTrainingTests.params,
        )

        print(loss, accuracy)

    def test_gradient_fn(self):
        dataloader = RMTrainingTests.init_train_dataloader()
        example_batch = next(dataloader)

        (loss, accuracy), gradients = grad_fn(
            RMTrainingTests.model.__call__,
            example_batch,
            RMTrainingTests.params,
        )

        optimizer = optax.adamw(LEARNING_RATE)
        opt_state = optimizer.init(RMTrainingTests.params)
        optimizer.update(gradients, opt_state, RMTrainingTests.params)

    def test_step_train_state(self):
        dataloader = RMTrainingTests.init_train_dataloader()
        example_batch = next(dataloader)

        optimizer = optax.adamw(LEARNING_RATE)
        opt_state = optimizer.init(RMTrainingTests.params)
        train_state = TrainState(
            0,
            RMTrainingTests.model.__call__,
            RMTrainingTests.params,
            optimizer,
            opt_state,
        )

        for _ in range(NUM_TRAIN_STEPS):
            train_state, (loss, accuracy) = jit_train_step(train_state, example_batch)
            print(accuracy, loss)
