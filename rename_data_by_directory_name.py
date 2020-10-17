import os
import glob

for root, dirs, files in os.walk(r"C:\Users\host\PycharmProjects\pythonProject\data_3\train"):
    for idx, f in enumerate(files):
        os.rename(os.path.join(root, f), os.path.join(root, root.split(os.path.sep)[-1] + "_" + str(idx) + ".jpg"))
