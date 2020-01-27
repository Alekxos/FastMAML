import os
from os import path

def create_directory(dir_path):
    if not path.exists(dir_path):
        os.mkdir(dir_path)