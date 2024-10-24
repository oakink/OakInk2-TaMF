import os
import yaml
from easydict import EasyDict


def load_cfg():
    cfg_filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "PointTransformer_8192point_2layer.yaml"))
    with open(cfg_filepath, 'r') as ifstream:
        res = yaml.load(ifstream, Loader=yaml.FullLoader)
    res = EasyDict(res)
    # use color
    res.use_color = True
    res.model.point_dims = 6
    # enable max_pool
    res.model.use_max_pool = True
    return res

def derive_cfg(cfg):
    use_max_pool = getattr(cfg.model, "use_max_pool", False)
    res = EasyDict()
    res.point_cloud_dim = cfg.model.point_dims
    res.backbone_output_dim = cfg.model.trans_dim if not use_max_pool else cfg.model.trans_dim * 2
    res.point_token_len = cfg.model.num_group + 1 if not use_max_pool else 1
    res.projection_hidden_layer = cfg.model.get('projection_hidden_layer', 0)
    res.use_max_pool = use_max_pool
    return res