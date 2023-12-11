import numpy as np
from typing import List, Dict, Any, Union
from peppy import Project
from peppy.const import SAMPLE_MODS_KEY, CONSTANT_KEY, CONFIG_KEY, NAME_KEY
from fastembed.embedding import FlagEmbedding as Embedding

import flatdict

from .utils import read_in_key_words
from .const import DEFAULT_KEYWORDS, MIN_DESCRIPTION_LENGTH


class PEPEncoder(Embedding):
    """
    Simple wrapper of the sentence trasnformer class that lets you
    embed metadata inside a PEP.
    """

    def __init__(self, model_name: str, keywords_file: str = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.keywords_file = keywords_file

        # read in keywords
        if self.keywords_file is not None:
            self.keywords: List[str] = read_in_key_words(self.keywords_file)
        else:
            self.keywords: List[str] = DEFAULT_KEYWORDS

    def mine_metadata_from_dict(
        self, project: Dict[str, any], min_desc_length: int = MIN_DESCRIPTION_LENGTH
    ) -> str:
        """
        Mine the metadata from a dictionary.

        :param project: A dictionary representing a peppy.Project instance.
        :param min_desc_length: The minimum length of the description.
        """
        # project_config = project.get(CONFIG_KEY) or project.get(
        #     CONFIG_KEY.replace("_", "")
        # )
        # fix bug where config key is not in the project,
        # new database schema does not have config key
        project_config = project
        if project_config is None:
            return ""
        if (
            SAMPLE_MODS_KEY not in project_config
            or CONSTANT_KEY not in project_config[SAMPLE_MODS_KEY]
        ):
            return project[NAME_KEY] or ""

        # project_level_dict: dict = project_config[SAMPLE_MODS_KEY][CONSTANT_KEY]
        # Flatten dictionary
        project_level_dict: dict = flatdict.FlatDict(project_config)
        project_level_attrs = list(project_level_dict.keys())
        desc = ""

        # search for "summary" in keys, if found, use that first, then pop it out
        # should catch if key simply contains "summary"
        for attr in project_level_attrs:
            if "summary" in attr:
                desc += str(project_level_dict[attr]) + " "
                project_level_attrs.remove(attr)
                break
            
        # build up a description using the rest
        for attr in project_level_attrs:
            if any([kw in attr for kw in self.keywords]):
                desc += str(project_level_dict[attr]) + " "

        # return if description is sufficient
        if len(desc) > min_desc_length:
            return desc
        else:
            return ""

    def mine_metadata_from_pep(
        self, project: Project, min_desc_length: int = MIN_DESCRIPTION_LENGTH
    ) -> str:
        """
        Mine the metadata from a peppy.Project instance. Small wrapper around
        the `mine_metadata_from_dict` method. It converts the peppy object to
        a dictionary and then calls the `mine_metadata_from_dict` method.

        :param project: A peppy.Project instance.
        :param min_desc_length: The minimum length of the description.
        """
        project_dict = project.to_dict(extended=True)
        return self.mine_metadata_from_dict(
            project_dict, min_desc_length=min_desc_length
        )
