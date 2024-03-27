import torch

#read video
import cv2
import os

file = r"C:\Users\jl255\Downloads\IMG_2925.mov"

# we will extract frames from the video and save them as images in the folder called video1
# we will save the frames in the folder video1
folder = "video8"
os.makedirs(folder, exist_ok=True)

# capture the video
cap = cv2.VideoCapture(file)
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{folder}/frame_{i}.jpg", frame)
    i += 1
cap.release()
cv2.destroyAllWindows()
