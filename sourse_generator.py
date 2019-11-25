import numpy as np
import cv2

for i in range(1500):
    start_x = np.random.randint(100, 200)
    start_y = np.random.randint(40, 200)

    line_size = np.random.randint(15, 40)

    end_x = start_x + np.random.randint(-70, 70)
    end_y = start_y +  np.random.randint(-70, 70)

    img = np.full((240,240,3), 1, dtype=np.float)
    cv2.line(img, (start_x,start_y), (end_x,end_y), (0,0,0), line_size)

    cv2.imwrite('./sourse_dataset/'+str(i)+'.png', img)