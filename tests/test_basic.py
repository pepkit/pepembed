import os
from logging import getLogger

import flatdict
import peppy
from peppy import Project

import pepembed
from pepembed.const import *
from pepembed.pepembed import PEPEncoder

_LOGGER = getLogger("pepembed")


class Testpepembed:
    def test_search(self):
        """Basic example of a test"""
        # hf_model = "sentence-transformers/all-MiniLM-L12-v2"   #this is the default in argsparser
        found = False
        keywordsfilepath = os.path.join(os.getcwd() + "/tests/data/keywordstest.txt")

        encoder = PEPEncoder(DENSE_ENCODER_MODEL, keywords_file=keywordsfilepath)

        p = peppy.Project(
            os.path.join(os.getcwd() + "/tests/data/testconfigs/testpep1.yaml")
        )
        p = p.to_dict(extended=True)

        d = encoder.mine_metadata_from_dict(p, min_desc_length=20)

        for k, v in flatdict.FlatDict(p["_config"]).items():
            if str(v) in d:
                found = True

        assert found == True
