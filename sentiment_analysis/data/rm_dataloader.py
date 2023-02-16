from typing import Callable, Iterator, Dict, Literal, Optional, NamedTuple, Tuple, List
import datasets
import jax


class ProcessedTextBatch(NamedTuple):
    """
    Default output from HF tokenizer isn't a valid JAX tree.
    """

    input_ids: jax.numpy.ndarray
    attention_mask: jax.numpy.ndarray
    labels: jax.numpy.ndarray


NumBatches = int


def get_dataloader(
    dataset: datasets.Dataset,
    tokenizer: Callable,
    batch_size: int,
    block_size: int,  # max num tokens per sequence
    num_epochs: float = 1.0,
    prng_key: Optional[jax.random.PRNGKeyArray] = None,
) -> Tuple[NumBatches, Callable[[], Iterator[ProcessedTextBatch]]]:
    """
    Yield pairwise examples.
    """
    assert not isinstance(
        dataset, datasets.DatasetDict
    ), 'Must select a particular dataset split (e.g., "train")'
    assert isinstance(dataset, datasets.Dataset)

    # Leftover items would be discarded
    num_batches = len(dataset) // batch_size
    num_batches_with_repeat = int((num_epochs * len(dataset)) // batch_size)
    indices = jax.numpy.arange(len(dataset))

    if prng_key is None:
        shuffled_indices = indices.tolist()
    else:
        shuffled_indices = jax.random.permutation(prng_key, indices).tolist()

    def tokenize(texts: List[str]) -> ProcessedTextBatch:
        tokenizer_output = tokenizer(
            texts,
            return_tensors="jax",
            max_length=block_size,
            padding="max_length",
            truncation=True,
        )

        return ProcessedTextBatch(
            input_ids=tokenizer_output.input_ids,
            attention_mask=tokenizer_output.attention_mask,
            labels=jax.numpy.full(tokenizer_output.input_ids.shape[0], -1),
        )

    def _initialize_dataloader() -> Iterator[ProcessedTextBatch]:
        for batch_index in range(num_batches_with_repeat):
            first = batch_size * (batch_index % num_batches)
            last = first + batch_size
            batch_indices = shuffled_indices[first:last]
            data = dataset[batch_indices]

            processed_batch = tokenize(data["sentence"])
            label_array = jax.numpy.array(data["label"])
            processed_batch = processed_batch._replace(labels=label_array)

            yield processed_batch

    return num_batches, _initialize_dataloader
