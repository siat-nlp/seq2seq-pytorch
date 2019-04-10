import yaml
import os
from trainer.trainer import train

filename = open('filename.txt', 'r', encoding='utf-8').readline().strip()
config = yaml.load(open(os.path.join('configs', filename)))
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['train']['gpu'])
train(config)