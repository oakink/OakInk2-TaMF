import logging


def suppress():
    logger = logging.getLogger("torch.nn.parallel.distributed")
    logger.setLevel(logging.WARNING)


# effective on import
suppress()
