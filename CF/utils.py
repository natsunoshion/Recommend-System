import os
from datetime import datetime
import pickle
import logging
import csv


def save_model(obj, path):
    try:
        with open(path, "wb") as file:
            pickle.dump(obj, file)
        logging.info(f"Model saved successfully at {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


def load_model(path):
    try:
        with open(path, "rb") as file:
            obj = pickle.load(file)
        logging.info(f"Model loaded successfully from {path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def setup_logging(log_dir, log_file_prefix):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_file_prefix}_{current_time}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a"), logging.StreamHandler()],
    )


def archive_existing_directory(base_dir):
    if os.path.exists(base_dir):
        archive_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = f"{base_dir}_{archive_time}"
        os.rename(base_dir, archive_dir)
        logging.info(f"Archived existing directory to {archive_dir}")


def file_read(path):
    if os.path.exists(path):
        file = open(path, "r", encoding="utf-8")
        csv_reader = csv.reader(file)
        next(csv_reader)
        return file, csv_reader
    else:
        print("ERROR: %s does not exit!" % (path,))


def file_save(file, path):
    with open(path, "wb") as f:
        pickle.dump(file, f)


def rate_modify(rate):
    if rate < 0:
        return 0
    elif rate > 100:
        return 100
    else:
        return round(rate)


def valid_rate(rate):
    if rate < 0:
        return 0
    elif rate > 100:
        return 100
    else:
        return rate
