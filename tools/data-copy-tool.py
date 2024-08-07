import pandas as pd
import glob
import shutil
import os


df = pd.read_csv('./data/labels.csv')
file_list = glob.glob("./data/images/*.bmp")

list_of_positive_images = df['Image']
empty_files = list(set(file_list) - set(list_of_positive_images))

print(len(file_list))
print(len(list_of_positive_images))
print(len(empty_files))

for file_name in empty_files:
    head, tail = os.path.split(file_name)
    shutil.move(file_name, './data/empty/' + tail)