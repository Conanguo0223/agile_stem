import wandb
from lineflow import __name__ as project_name


class Logger(object):
    """
    Handles logging

    Args:
        type (string): Can either be tb (tensorboard) or wandb (Weights & Biases). In case
        you use wandb please refer to (https://docs.wandb.ai/quickstart) to get startet.
    """
    def __init__(self, project_name=project_name, config=None, line_name=None):
        wandb.init(project=project_name, config=config, tags=[line_name])

    def log(self, data, step):
        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()
