import os
import json
import csv
import tempfile
import click
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType

def get_client(host, port, username, password):
    # If host already contains a colon, assume port was specified
    server = f"{host}:{port}" if ":" not in host else host
    return make_client(host=server, port=port, credentials=(username, password))

@click.group()
def cvat():
    """A CLI for importing/exporting datasets in CVAT."""
    pass

@cvat.command()
def test_conn():
    with get_client(host="http://localhost", port=8080, username="john.doe", password="Test_1234") as client:
        print("CLIENT", client)

@cvat.command()
def import_coco_dataset():
    """
    Create a new CVAT task from a COCO dataset.
    
    This command uploads all images found in IMAGES_DIR to a new task named TASK_NAME,
    then imports COCO annotations from COCO_JSON. Optionally, label definitions can be read
    from LABELS_FILE (a JSON file).
    """
    host = "http://localhost"
    port=8080
    username="john.doe"
    password="Test_1234"
    task_name="spine"
    coco_json="data/7s/annotations/instances_val2017.json"
    with open(coco_json) as fp:
        content=json.load(fp)
    print(content["categories"])
    #exit()
    images_dir="data/7s/val2017"
    labels = [{
        "name": category["name"],
        "color": "#ff0000",
        "attributes": [{
            "name": "a",
            "mutable": True,
            "input_type": "number",
            "default_value": "5",
            "values": ["4", "5", "6"],
        }]
    } for category in content["categories"]]
    
    with get_client(host, port, username, password) as client:
        
        task_spec = {"name": task_name, "labels": labels}

        # List image files from the given directory (supported extensions)
        supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        images = [
            os.path.join(images_dir, fname)
            for fname in os.listdir(images_dir)
            if fname.lower().endswith(supported_ext)
        ]
        if not images:
            click.echo("No images found in the specified images directory.")
            return

        click.echo(f"Creating task '{task_name}' with {len(images)} images...")
        task = client.tasks.create_from_data(
            spec=task_spec,
            resource_type=ResourceType.LOCAL,
            resources=images
        )
        click.echo(f"Task created with id: {task.id}")

        # Import annotations from the COCO JSON file
        click.echo("Importing COCO annotations...")
        # (Assumes that the task object has an import_annotations() method.)
        task.import_annotations(format_name="COCO 1.0", filename=coco_json)
        click.echo("COCO dataset imported successfully.")

@cvat.command()
@click.option('--host', default='localhost', help='CVAT server host.')
@click.option('--port', default='8080', help='CVAT server port.')
@click.option('--username', prompt=True, help='CVAT username')
@click.option('--password', prompt=True, hide_input=True, help='CVAT password')
@click.option('--task-id', prompt=True, type=int, help='Existing CVAT task id to add detections')
@click.option('--csv-file', type=click.Path(exists=True), prompt=True, help='Path to CSV file with detections')
def import_detections(host, port, username, password, task_id, csv_file):
    """
    Import object detection results from a CSV file into an existing CVAT task.
    
    The CSV file is expected to have at least the following columns:
      image, label, x, y, width, height, and optionally score.
    The command converts the CSV into a temporary COCO-format JSON and uploads it.
    """
    annotations = []
    images_dict = {}  # mapping from image file name to unique image id
    ann_id = 1

    # Read CSV file and build COCO-style annotation structure
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name = row['image']
            # For simplicity, we use a fixed category id (e.g. 1).
            # In a more advanced implementation, you might map different labels.
            category_id = 1  
            try:
                x = float(row['x'])
                y = float(row['y'])
                width = float(row['width'])
                height = float(row['height'])
            except ValueError:
                continue  # Skip rows with invalid numbers
            score = float(row['score']) if 'score' in row and row['score'] else 1.0

            if image_name not in images_dict:
                images_dict[image_name] = len(images_dict) + 1
            image_id = images_dict[image_name]

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
                "score": score,
                "segmentation": []
            }
            annotations.append(ann)
            ann_id += 1

    # Build the COCO JSON structure
    coco_data = {
        "images": [
            {"id": img_id, "file_name": fname}
            for fname, img_id in images_dict.items()
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "detection"}]
    }

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmpfile:
        json.dump(coco_data, tmpfile)
        tmpfile_path = tmpfile.name

    with get_client(host, port, username, password) as client:
        task = client.tasks.retrieve(task_id)
        click.echo(f"Uploading detections from '{csv_file}' into task {task_id}...")
        task.import_annotations(tmpfile_path, format="COCO 1.1")
        click.echo("Detections imported successfully.")

    os.remove(tmpfile_path)

@cvat.command()
@click.option('--host', default='localhost', help='CVAT server host.')
@click.option('--port', default='8080', help='CVAT server port.')
@click.option('--username', prompt=True, help='CVAT username')
@click.option('--password', prompt=True, hide_input=True, help='CVAT password')
@click.option('--task-id', prompt=True, type=int, help='CVAT task id to export')
@click.option('--format', default="COCO 1.1", help='Export format (default: "COCO 1.1")')
@click.option('--output', type=click.Path(), prompt=True, help='Output file path (e.g., output.zip)')
def export_coco_dataset(host, port, username, password, task_id, format, output):
    """
    Export an updated CVAT task dataset in COCO format.
    
    This command downloads the dataset from the task with id TASK_ID in the given
    format and writes it to the specified output file.
    """
    with get_client(host, port, username, password) as client:
        task = client.tasks.retrieve(task_id)
        click.echo(f"Exporting task {task_id} in format '{format}' to '{output}'...")
        # (Assumes that task.export_dataset() returns a binary stream.)
        export_response = task.export_dataset(format=format)
        with open(output, "wb") as out_file:
            out_file.write(export_response.read())
        click.echo("Dataset exported successfully.")

if __name__ == '__main__':
    cli()
