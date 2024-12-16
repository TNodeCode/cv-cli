import click
from src.mmdet.utils import MMDetModels

@click.group()
def cli():
    """CLI for training and managing a YOLO model on a custom dataset."""
    pass

@cli.command()
def train():
    """Train the YOLO model."""
    print("Start training")
    MMDetModels.train()
    print("Training completed.")

if __name__ == '__main__':
    cli()
