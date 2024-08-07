import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

REGION_NUMBER = 10
INTERNAL_RADIUS = 160
EXTERNAL_RADIUS = 620
ANGLE_STEP = 360/REGION_NUMBER

csv_file = open('./data/labels.csv', "w")
csv_file.write('Image,Label\n')

file_list = glob.glob("./data/images/*.bmp")

file_list.sort()

file_list = []
file_list.append('./data/images/1128.bmp')

for file_name in file_list:
    img = cv.imread(file_name, cv.IMREAD_COLOR)
    if img is None:
        print("file could not be read->", file_name)
        continue

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv.circle(mask, (640, 640), 640 , 255, -1)
    img = cv.bitwise_and(img, img, mask=mask)
    
    cv.circle(img,(640,640), INTERNAL_RADIUS, (255,0,0),1)
    cv.circle(img,(640,640), EXTERNAL_RADIUS, (255,0,0),1)

    for i in range(REGION_NUMBER):
         angle = (ANGLE_STEP * i) * np.pi / 180
         in_x = 640 + (INTERNAL_RADIUS * np.cos(angle)).astype('int')
         in_y = 640 + (INTERNAL_RADIUS * np.sin(angle)).astype('int')
         ex_x = 640 + (EXTERNAL_RADIUS * np.cos(angle)).astype('int')
         ex_y = 640 + (EXTERNAL_RADIUS * np.sin(angle)).astype('int')
         cv.line(img, (in_x, in_y), (ex_x, ex_y), (255,0,0) , 1)

    fig = plt.figure(figsize=[20,20])
    ax = fig.add_subplot(111)
    ax.set_title(file_name)
    ax.imshow(img)

    region = -1

    def onclick(event):
        global region
        ix, iy = event.xdata - 640, event.ydata - 640
        print ('x = ', ix , ' ,y = ', iy)
        len = np.linalg.norm([ix,iy])
                             
        if len.astype('int') < INTERNAL_RADIUS:
            region = 0
        elif len.astype('int') > EXTERNAL_RADIUS:
            region = REGION_NUMBER + 1
        else:
            v = [ix,iy]/len
            angle = np.arccos(v[0]) * 180 / np.pi
            print('angle=', angle)
            n = (angle / ANGLE_STEP).astype('int')
            if v[1] > 0:
                region = 1 + n
            else :
                region = (REGION_NUMBER) - n
        print(region)
        

    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event))

    while plt.waitforbuttonpress(0) != None:
        if region != -1:
            text = file_name+','+str(region)+'\n'
            csv_file.write(text)
        break
    plt.close()
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite('./pictures/regions.png', img=img)

csv_file.close()