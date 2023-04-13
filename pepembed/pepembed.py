import numpy as np
from typing import List, Dict, Any, Union
from peppy import Project
from peppy.const import SAMPLE_MODS_KEY, CONSTANT_KEY, CONFIG_KEY, NAME_KEY
from sentence_transformers import SentenceTransformer


from .utils import read_in_key_words
from .const import DEFAULT_KEYWORDS, MIN_DESCRIPTION_LENGTH


class PEPEncoder(SentenceTransformer):
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
        project_config = project.get(CONFIG_KEY) or project.get(CONFIG_KEY.replace("_", ""))
        if project_config is None:
            return ""
        if (
            SAMPLE_MODS_KEY not in project_config
            or CONSTANT_KEY not in project_config[SAMPLE_MODS_KEY]
        ):
            return project[NAME_KEY] or ""

        project_level_dict: dict = project_config[SAMPLE_MODS_KEY][CONSTANT_KEY]
        project_level_attrs = list(project_level_dict.keys())
        desc = ""

        # build up a description
        for attr in project_level_attrs:
            if any([kw in attr for kw in self.keywords]):
                desc += project_level_dict[attr] + " "

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

    def embed(
        self, projects: Union[dict, List[dict], Project, List[Project]], **kwargs
    ) -> np.ndarray:
        """
        Embed a PEP based on it's metadata.

        :param projects: A PEP or list of PEPs to embed.
        :param kwargs: Keyword arguments to pass to the `encode` method of the SentenceTransformer class.
        """
        # if single dictionary is passed
        if isinstance(projects, dict):
            desc = self.mine_metadata_from_dict(projects)
            return super().encode(desc, **kwargs)

        # if single peppy.Project is passed
        elif isinstance(projects, Project):
            desc = self.mine_metadata_from_pep(projects)
            return super().encode(desc, **kwargs)

        # if list of dictionaries is passed
        elif isinstance(projects, list) and isinstance(projects[0], dict):
            descs = [self.mine_metadata_from_dict(p) for p in projects]
            return super().encode(descs, **kwargs)

        # if list of peppy.Projects is passed
        elif isinstance(projects, list) and isinstance(projects[0], Project):
            descs = [self.mine_metadata_from_pep(p) for p in projects]
            return super().encode(descs, **kwargs)

        # else, return ValueError
        else:
            raise ValueError(
                "Invalid input type. Must be a dictionary, peppy.Project, list of dictionaries, or list of peppy.Projects."
            )
