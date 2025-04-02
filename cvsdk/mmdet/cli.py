import click
from cvsdk.mmdet.utils import MMDetModels

@click.group()
def mmdet():
    """CLI for training and managing a MMDet model on a custom dataset."""
    pass

@mmdet.command()
@click.argument("config-file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("load-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def train(config_file: str, load_from: str | None):
    """Train the MMDet model."""
    MMDetModels.train(config_file=config_file, load_from=load_from)


@mmdet.command()
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-file", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--load-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def extract_backbone(config_file: str, output_file: str, load_from: str | None):
    MMDetModels.extract_backbone(
        config_file=config_file,
        load_from=load_from,
        output_file=output_file
    )


@mmdet.command()
@click.option("--source-config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--target-config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-file", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--load-source-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def copy_backbone(source_config_file: str, target_config_file: str, output_file: str, load_source_from: str | None):
    MMDetModels.copy_backbone(
        source_config_file=source_config_file,
        target_config_file=target_config_file,
        load_source_from=load_source_from,
        output_file=output_file
    )