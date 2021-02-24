# -*- coding: utf-8 -*-

import json
import os

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary)
    """
    if not os.path.exists(json_file):
        return {"piece_size": 0.2, "class_type": "3C"}

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict
