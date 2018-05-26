import os
import cv2
import numpy as np
from shapes import Rectangle, Point
from utils import fill_zero, find_bounding_box, coords_within_boundary, get_foreground
from sklearn.decomposition import PCA
from Threshold import BaseThreshold, NaiveThreshold, DecisionTreeThreshold
from PIL import Image


# noinspection PyBroadException
class Alarm:
    def __init__(self, video_path: str, rectangles: (tuple, list), threshold):
        # store input video path
        self.video_path = video_path
        # self.image_path = video_path.split('/')[-1]
        # force_mkdir(os.path.join('.',self.image_path))
        self.threshold = threshold  # type: BaseThreshold
        self.alarm_list = []
        self.pca_solver = PCA(n_components=1)
        assert os.path.isfile(self.video_path), "No such file: %s" % video_path
        cap = cv2.VideoCapture(self.video_path)
        # obtain background image
        ret, background = cap.read()
        cap.release()
        self.background = background if ret else np.zeros((480, 640, 3), dtype=np.uint8)  # type: np.ndarray
        self.display_background = np.copy(background)
        self.pixel_count = self.background.shape[0] * self.background.shape[1]

        # define a list of rectangular areas, where alarms are muted
        upper, lower, left, right = 0, self.background.shape[0], 0, self.background.shape[1]
        background_rectangle = Rectangle(upper, lower, left, right)
        self.mute_rects = rectangles
        for rectangle in rectangles:  # type: Rectangle
            assert rectangle.is_contained_in(background_rectangle), 'rectangle not contained in background'
        print("all input rectangles for mute area are legal")
        self.parameter_list = list()

    def _process_frame(self, frame, frame_index, pixel_diff_threshold, pixel_count_threshold, beta):
        # HACK naive implementation of background subtraction
        foreground = get_foreground(frame, self.background, pixel_diff_threshold)
        display_foreground = get_foreground(frame, self.display_background, pixel_diff_threshold)
        # update background with running average
        self.display_background = self.display_background * beta + frame * (1. - beta)

        # get the two opposite vertices of real bounding box
        xx, yy = np.nonzero(foreground)
        upper, lower, left, right = find_bounding_box(xx, yy, pix_count_thres=pixel_count_threshold)
        vertices = ((left, upper), (right, lower))

        # check whether the foreground is located in mute areas
        for rect in self.mute_rects:
            fill_zero(foreground, rect)

        # get the basic stats of foreground with muted areas (points_count, value, area, ratio, obliqueness)
        value = np.sum(foreground)
        xx, yy = np.nonzero(foreground)
        u_l_l_r = find_bounding_box(xx, yy, pix_count_thres=pixel_count_threshold)  # returns upper, lower, left, right
        bounding_rectangle = Rectangle(*u_l_l_r)
        area = bounding_rectangle.get_area()
        ratio = bounding_rectangle.get_ratio()
        try:
            coords = coords_within_boundary(xx, yy, *u_l_l_r, zero_mean=True)
            self.pca_solver.fit(coords)
            obliqueness = self.pca_solver.explained_variance_ratio_
        except:
            obliqueness = 0.5

        # check whether the shape of foreground should trigger the alarm
        self.parameter_list.append((len(xx), value, area, ratio, obliqueness))
        if self.threshold.check(len(xx), value, area, ratio, obliqueness):
            cv2.rectangle(frame, *vertices, color=(0, 0, 255))
            cv2.putText(frame, "WARNING", (left, upper), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            self._trigger_alarm(frame_index, frame)
        else:
            cv2.rectangle(frame, *vertices, color=(255, 0, 0))
            cv2.putText(frame, "Object", (left, upper), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))
        return frame, display_foreground

    def process_video(self, pixel_diff_threshold=40, pixel_count_threshold=4000, beta=1.0):
        self.alarm_list = []
        cap = cv2.VideoCapture(self.video_path)
        frame_index = 0
        cv2.namedWindow('frame')
        # cv2.namedWindow("foreground")
        while cap.isOpened():
            frame_index += 1
            ret, frame = cap.read()
            # XXX the second part of the condition is necessary, otherwise cv2.imshow() shows nothing
            if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            frame, display_foreground = self._process_frame(frame, frame_index, pixel_diff_threshold,
                                                            pixel_count_threshold, beta)
            for rect in self.mute_rects:  # type: Rectangle
                upper, lower, left, right = rect.get_upper(), rect.get_lower(), rect.get_left(), rect.get_right()
                cv2.rectangle(frame, (left, upper), (right, lower), color=(0, 255, 0))
            cv2.imshow('frame', frame)
            # cv2.imshow('foreground', display_foreground)

        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    def _trigger_alarm(self, frame_index, frame):
        print("Alarm at frame %d" % frame_index)
        # img = Image.fromarray(frame)
        # img.save(os.path.join(self.image_path, "warning_frame_%d.jpg" % frame_index))
        self.alarm_list.append(frame_index)


if __name__ == "__main__":
    rectangles = list()
    rectangles.append(Rectangle(0, 480, 0, 162))  # sofa
    rectangles.append(Rectangle(125, 300, 162, 214))  # sofa
    rectangles.append(Rectangle(52, 143, 505, 640))  # desk
    rectangles.append(Rectangle(208, 292, 494, 603))  # chair
    # rectangles.append(Rectangle(94, 140, 596, 630))  # cup
    rectangles.append(Rectangle(0, 103, 291, 392))  # mirror
    rectangles.append(Rectangle(64, 129, 242, 294))  # trash can

    video_path = '/Users/liushuheng/Desktop/falter-data/3.wmv'
    # threshold = NaiveThreshold(point_count=11500, value=0, area=15000, ratio=0, obliqueness=0.85)
    threshold = DecisionTreeThreshold()
    alarm = Alarm(video_path, rectangles, threshold)
    alarm.process_video(pixel_diff_threshold=40, pixel_count_threshold=4000, beta=0.999)

    # with open('/Users/liushuheng/Desktop/DecisionTreeInput1.csv', 'w') as f:
    #     for tup in alarm.parameter_list:
    #         f.write("%d,%d,%d,%f,%f\n" % tup)

    # for index in alarm.alarm_list: print(index)
