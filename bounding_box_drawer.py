# %%
import cv2
import numpy as np
import os
import pandas as pd

# get the list of files in the folder
folder = "video1"
csv_file = "labels_my-project-name_2024-03-23-07-10-18.csv"

# read the csv file
csv_file = pd.read_csv(csv_file)

#%%
for i in range(len(csv_file)):
    img = cv2.imread(f"{folder}/{csv_file['image_name'][i]}")
    # draw the bounding box
    x1 = int(csv_file['bbox_x'][i])
    y1 = int(csv_file['bbox_y'][i])
    x2 = x1 + int(csv_file['bbox_width'][i])
    y2 = y1 + int(csv_file['bbox_height'][i])
    print(x1, y1, x2, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # show the image
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break