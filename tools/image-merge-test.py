import glob
from PIL import Image, ImageDraw
import random
import numpy as np


background_image_list = glob.glob("./data/empty/*.bmp")
object_image_list = glob.glob("./data/objects/*.bmp")

background_num = len(background_image_list)
objects_num = len(object_image_list)

background_path = random.choice(background_image_list)

background = Image.open(background_path)
label = 11
add_image_n = random.randint(0, 9)
add_image = (add_image_n != 0)

if add_image:
    object_path = random.choice(object_image_list)
    image = Image.open(object_path)

    rotation_angle = random.randrange(0, 360)
    image = image.rotate(rotation_angle)
    scale_x = random.randrange(5,10)/10
    scale_y = random.randrange(5,10)/10
    image = image.resize((int(scale_x * image.size[0]), int(scale_y * image.size[1])))

    R = random.randrange(0, 500)
    A = random.randrange(0, 360)
    A_RAD = A / 180 * np.pi

    x = 640 + int(R * np.cos(A_RAD))
    y = 640 + int(R * np.sin(A_RAD))

    if R < 160 :
        label = 0
    elif R > 620:
        label = 11
    else :
        label = int(A / 36) + 1

    x_shift, y_shift = image.size[0]//2, image.size[1]//2
    
    background.paste(image, (x-x_shift, y-y_shift), image)

# ImageDraw.Draw(background)
background.show()
print(label)
