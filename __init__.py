import os

filepath = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(filepath)
DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets/')

index_path = os.path.join(PROJECT_ROOT,'datasets/index.csv')
test_path = os.path.join(PROJECT_ROOT,'datasets/0826B')
rnn_test_path = os.path.join(PROJECT_ROOT,'datasets/0826C')

