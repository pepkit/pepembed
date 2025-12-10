from typing import List, Dict, Generator, Any
import re
import flatdict


from .const import DEFAULT_KEYWORDS


def read_in_key_words(key_words_file: str) -> List[str]:
    """Read in key words from a file."""
    key_words = []
    with open(key_words_file, "r") as f:
        for line in f:
            key_words.append(line.strip())
    return key_words


def batch_generator(iterable, batch_size) -> Generator[Any, Any, None]:
    """Batch generator."""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


def markdown_to_text(md: str) -> str:
    # Remove markdown links: [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    # Remove any other markdown markup if needed (bold, italics, etc.)
    text = re.sub(r"[*_`]", "", text)
    return text


def mine_metadata_from_dict(
    project: Dict[str, any],
    description: str = "",
    name: str = "",
    keywords: List[str] = DEFAULT_KEYWORDS,
) -> str:
    """
    Mine the metadata from a dictionary.

    :param project: A dictionary representing a peppy.Project instance.
    :param description: An optional description to include.
    :param name: An optional name to include.
    :param keywords: A list of keywords to search for in the metadata.

    """

    project_config = project
    if project_config is None:
        return ""

    # Flatten dictionary
    project_level_dict = flatdict.FlatDict(project_config)
    project_level_attrs = list(project_level_dict.keys())
    desc = ""

    # # search for "summary" in keys, if found, use that first, then pop it out
    # # should catch if key simply contains "summary"
    # for attr in project_level_attrs:
    #     if "summary" in attr:
    #         desc += str(project_level_dict[attr]) + " "
    #         project_level_attrs.remove(attr)
    #         break

    # build up a description using the rest
    for attr in project_level_attrs:
        if any([kw in attr for kw in keywords]):
            desc += str(project_level_dict[attr]) + " "

    if name and description:
        return f"Name: {name}. Description: {description}. Metadata: {desc.strip()}"
    return desc.strip()
