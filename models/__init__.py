"""
@Time : 2024/11/24 13:57
@Auth ： Weiming
@github : https://github.com/Weimingai
@Blog : https://www.cnblogs.com/weimingai/
@File ：coco.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

from .el_detr import build


def build_model(args):
    return build(args)
