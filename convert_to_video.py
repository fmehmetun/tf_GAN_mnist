import cv2
import numpy as np
import os

target_dir = "outs"

os.chdir(target_dir)

frames_file = os.listdir(".")
frames_file = sorted([int(f.replace(".jpg", "")) for f in frames_file])
frames_file = [str(f)+".jpg" for f in frames_file]

out = cv2.VideoWriter("../"+str(target_dir)+".mp4", cv2.VideoWriter_fourcc(*"H264"), 30, cv2.imread(frames_file[0]).shape[:-1])

for f in range(len(frames_file)):
	frame = cv2.imread(frames_file[f])
	out.write(frame)

	print(f, len(frames_file))

out.release()