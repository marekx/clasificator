import os
import glob

for root, dirs, files in os.walk(r"C:\Praca inżynierska\marekx\clasificator\FF_2.0\original dataset\test"):
    for idx, f in enumerate(files):
        os.rename(os.path.join(root, f), os.path.join(root, root.split(os.path.sep)[-1] + "." + str(idx) + ".jpg"))
