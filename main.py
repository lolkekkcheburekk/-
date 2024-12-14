import cv2
import numpy as np
import time
import easyocr

nom = set()

faceCascade = cv2.CascadeClassifier('haara/haarcascade_russian_plate_number.xml')
video_capture = cv2.VideoCapture(0)
vd = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plaques = faceCascade.detectMultiScale(gr, 1.1, 5)
    for i, (x, y, w, h) in enumerate(plaques):
        roi_color = frame[y:y + h, x:x + w]
        r = 400.0 / roi_color.shape[1]
        dim = (400, int(roi_color.shape[0] * r))
        resized = cv2.resize(roi_color, dim, interpolation=cv2.INTER_AREA)
        imp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(imp, cv2.COLOR_RGB2GRAY)
        gup = cv2.GaussianBlur(gray, (5, 5), 0)
        bf = cv2.bilateralFilter(gup, 11, 17, 17)
        text = easyocr.Reader(['en'])
        res = text.readtext(bf, allowlist='ETYOPAHKXCBM0123456789', detail=0)
        if len(res) > 0:
            res = res[0]
            if len(res) > 7:
                if res[0] in 'ETYOPAHKXCBM' and res[1] in '0123456789' and res[1] in '0123456789' and res[
                    2] in '0123456789' and res[3] in '0123456789' and res[4] in 'ETYOPAHKXCBM' and res[
                    -1] in '011234566789':
                    nom.add(res)
        print(res, nom)
