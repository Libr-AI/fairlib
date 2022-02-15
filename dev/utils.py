import sys
sys.path.append('../')

from src import utils
import logging

log_file = r"D:\Project\Fair_NLP_Classification\logs\test.log"

utils.logging.configure(log_file, "INFO", prefix="")

logging.info('Base directory is {}'.format("asd"))