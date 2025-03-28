import click
from cvsdk.mmdet.utils import MMDetModels

@click.group()
def mmdet():
    """CLI for training and managing a MMDet model on a custom dataset."""
    pass

@mmdet.command()
@click.argument("config-file", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument("load-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=True))
def train(config_file: str, load_from: str | None):
    """Train the MMDet model."""
    MMDetModels.train(config_file=config_file, load_from=load_from)
