import os.path as osp

def source_path(__f__):
    return osp.join(osp.dirname(osp.abspath(__f__)))