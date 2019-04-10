import os
import yaml
from data_process.data_process import data_process

filename = open('filename.txt', 'r', encoding='utf-8').readline().strip()
config = yaml.load(open(os.path.join('configs', filename)))
data_process(config)