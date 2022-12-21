from typing import List


def read_in_key_words(key_words_file: str) -> List[str]:
    """Read in key words from a file."""
    key_words = []
    with open(key_words_file, "r") as f:
        for line in f:
            key_words.append(line.strip())
    return key_words


def batch_generator(iterable, batch_size) -> List:
    """Batch generator."""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]
