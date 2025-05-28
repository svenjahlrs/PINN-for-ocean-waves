from csv import DictWriter
import torch.nn as nn
import torch
import numpy as np
from csv import writer

def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)


def write_csv_line(path: str, line):
    """ writes a new line to a csv file at path"""
    with open(path, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(line)