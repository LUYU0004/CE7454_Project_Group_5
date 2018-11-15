import cv2
import sys
import os.path
from glob import glob

def detect(filename, cascade_file="data/anime/pretrained_model/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale3(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(48, 48),
                                     outputRejectLevels = True)
    for i in range(len(faces[0])):
        if faces[2][i] < 2:
            continue
        x, y, w, h = faces[0][i]
        face = image[y: y + h, x:x + w, :]
        # face = cv2.resize(face, (96, 96))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("data/anime/faces/" + save_filename, face)
