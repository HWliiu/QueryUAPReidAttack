from yacs.config import CfgNode


def get_default_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a fastreid CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()
