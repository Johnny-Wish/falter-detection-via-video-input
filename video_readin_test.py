import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import fill_zero, find_bounding_box, coords_within_boundary
from shapes import Rectangle

if __name__ == "__main__":
    cap = cv2.VideoCapture('/Users/liushuheng/Desktop/falter-data/1.wmv')
    ret, background = cap.read()
    frame_count = 1
    points, values, areas, ratios, obliques = [], [], [], [], []
    # beta = 0.995
    beta = 1.
    rectangles = list()
    rectangles.append(Rectangle(0, 480, 0, 162))  # sofa
    rectangles.append(Rectangle(125, 300, 162, 214))  # sofa
    # rectangles.append(Rectangle(94, 140, 596, 630))  # cup
    pca_solver = PCA(n_components=1)

    cv2.namedWindow("frame")
    cv2.startWindowThread()

    while cap.isOpened():
        ret, frame = cap.read()
        if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        frame_count += 1
        # cv2.imshow('frame', frame)

        # region foreground
        threshold = 50
        foreground = np.abs(frame.astype(np.int32) - background.astype(np.int32)).astype(np.uint8)
        foreground[foreground < threshold] = 0
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

        fill_zero(foreground, rectangles)
        xx, yy = np.nonzero(foreground)
        value_sum = np.sum(foreground)
        points.append(len(xx))
        values.append(value_sum)
        try:
            assert len(xx) > 6000
            upper, lower, left, right = find_bounding_box(xx, yy)
            # whether the following line is correct
            coords = coords_within_boundary(xx, yy, upper, lower, left, right, zero_mean=True)
            pca_solver.fit(coords)
            oblique = pca_solver.explained_variance_ratio_
        except:
            upper, lower, left, right = 0, 0, 0, 0
            oblique = 0.5
        obliques.append(oblique)

        area = (lower - upper) * (right - left)
        areas.append(area)
        try:
            ratio = float(lower - upper) / float(right - left)
            ratio = max(ratio, 1. / ratio)
        except ZeroDivisionError:
            ratio = 1.
        ratios.append(ratio)
        cv2.rectangle(frame, (left, upper), (right, lower), color=255)
        cv2.putText(frame, "Object", (left, upper), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))

        cv2.imshow("frame", frame)

        background = background * beta + frame * (1 - beta)
        # background = np.ceil(backgroundrc)

        # endregion

        # plot images
        # fig = plt.figure(figsize=(15, 6))
        # plt.imshow(foreground)
        # plt.axis('off')
        # plt.show()
        # break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # cv2.imshow('frame', gray)

    cap.release()

    with open('/Users/liushuheng/Desktop/stats.txt', 'w') as f:
        for count, val, area, ratio, oblique in zip(points, values, areas, ratios, obliques):
            f.write("%d %d %d %f %f\n" % (count, val, area, ratio, oblique))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
