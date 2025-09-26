import yaml
from box import ConfigBox



def load_params(params_file):
    with open(params_file,"r") as f:
        params=yaml.safe_load(f)
        ## access data example : dataset_url=params.data_load.dataset_url
        params=ConfigBox(params)
    return params