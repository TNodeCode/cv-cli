import os
import glob
import time
import numpy as np
import pandas as pd
from mmdet.apis import inference_detector, init_detector
from structlog import get_logger
from tqdm import tqdm

logger = get_logger()


def detect(
        config_file: str,
        epoch: int,
        results_file: str,
        dataset_dir: str,
        image_files: str,
        batch_size: int = 8,
        score_threshold: float = 0.5,
        device: str = "cuda:0",
        work_dir: str = "./work_dirs"
):
    config_file_path = f"{work_dir}/{config_file}"
    if epoch > 0:
        epochs = [epoch]
    else:
        epochs = range(1, len(glob.glob(f"{work_dir}/epoch_*.pth"))+1)

    logger.info(
        "Configuration",
        config_file_path=config_file_path,
        epoch=epoch,
        device=device,
        batch_size=batch_size
    )

    for epoch, weight_file in enumerate(epochs):
        logger.info(f"Detections for epoch {epoch} ...")
        weght_file = f"{work_dir}/epoch_{epoch}.pth"
        # Initialize the model
        logger.info("Loading model ...")
        model = init_detector(
            config_file_path,
            weight_file,
            device=device
        )
        logger.info("Model loaded")

        image_file_path = os.path.join(dataset_dir, image_files)
        logger.info("Image files", image_file_path=image_file_path)
        filenames = glob.glob(image_file_path)
        n_files = len(filenames)
        logger.info(f"Found {n_files} images")
        if n_files == 0:
            return
        n_batches=(n_files // batch_size) + 1

        detected_bboxes = []
        durations = []
        pbar = tqdm(range((n_files // batch_size) + 1))
        for b in pbar:
            pbar.set_description(f"Processing batch {b}/{n_batches} ...")
            if (len(filenames[b*batch_size:(b+1)*batch_size])) < 1:
                continue
            # Send images through model
            start = time.time()
            results = inference_detector(
                model=model,
                imgs=filenames[b*batch_size:(b+1)*batch_size]
            )
            end = time.time()
            durations.append(end - start)
            # Iterate over image results
            for i, result in enumerate(results):
                filename = filenames[b*batch_size+i].replace(os.sep, '/').replace(dataset_dir, "")
                bboxes = result.pred_instances.bboxes
                labels = result.pred_instances.labels
                scores = result.pred_instances.scores
                # Iterate over detected bounding boxes
                for ((x0, y0, x1, y1), label, score) in zip(bboxes, labels, scores):
                    if score >= score_threshold:
                        detected_bboxes.append({
                            "epoch": epoch,
                            "filename": filename,
                            "class_index": int(label),
                            "class_name": "spine",
                            "xmin": int(x0),
                            "ymin": int(y0),
                            "xmax": int(x1),
                            "ymax": int(y1),
                            "score": float(score)
                        })
        del model # free memory

    # Create the CSV file that contains the detections
    csv_filename = f"{work_dir}/{results_file}"
    os.makedirs(os.path.split(csv_filename)[0], exist_ok=True)
    pd.DataFrame(detected_bboxes).to_csv(csv_filename, index=False)
    logger.info("Saved CSV file at", csv_filename=csv_filename)

    # Print some statistics about the detection process
    durations = np.array(durations)
    logger.info(f"Inference took {durations.mean()}, per batch on average, std={durations.std()}")
    logger.info(f"Inference took {durations.mean() / batch_size} per image on average, std={durations.std() / batch_size}")