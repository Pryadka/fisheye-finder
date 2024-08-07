import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw
import numpy as np
import os

fish_model = tf.keras.models.load_model('result-best/model.keras')
# fish_model = tf.keras.models.load_model('results/08061704/model.keras')

fish_model.summary()

file_names = []
file_names.append('./data/images/575.bmp')
file_names.append('./data/images/1843.bmp')
file_names.append('./data/images/960.bmp')
file_names.append('./data/images/1118.bmp')
file_names.append('./data/images/1130.bmp')
file_names.append('./data/images/1131.bmp')
file_names.append('./data/images/1709.bmp')
file_names.append('./data/images/1716.bmp')
file_names.append('./data/images/1839.bmp')
file_names.append('./data/images/1842.bmp')
file_names.append('./data/images/1889.bmp')
file_names.append('./data/images/1892.bmp')
file_names.append('./data/images/1911.bmp')
file_names.append('./data/images/1918.bmp')


for file_name in file_names:
    img = Image.open(file_name)
    test_img = np.expand_dims(np.asarray(img), axis=0)

    result = fish_model.predict(test_img)

    print(result)

    region_class = np.argmax(result)

    draw = ImageDraw.Draw(img)
    if region_class == 0:
        draw.ellipse((640-160, 640-160, 640+160, 640+160), outline=(255,0,0))
    elif region_class < 11:
        angles = (region_class-1)*36, (region_class)*36
        draw.arc((640-160, 640-160, 640+160, 640+160), angles[0], angles[1], fill=(255,0,0))
        for angle in angles:
            angle = angle/180*np.pi
            y_s = np.sin(angle)
            x_s = np.cos(angle)
            draw.line((640+x_s*160, 640+y_s*160, 640+x_s*600, 640+y_s*600),fill=(255,0,0))

    head, tail = os.path.split(file_name)
    img.save('./pictures/pred_' + tail)