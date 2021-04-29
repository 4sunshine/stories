# from protobuf_to_dict import protobuf_to_dict
import mediapipe as mp
import numpy as np
import cv2

# mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class Hands:
    OPEN_THRESHOLD = 0.93
    STRAIGHT_THRESHOLD = 2.04

    def __init__(self, min_det=0.4, min_track=0.5, width=1280, height=720):
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=min_det, min_tracking_confidence=min_track)
        self.results = None
        self.width = width
        self.height = height
        self.hands_side = {'Right': np.array([]),
                           'Left': np.array([])}
        self.visible = {'Right': False,
                        'Left': False}
        self.palm_ind = [0, 5, 9, 13, 17]
        self.outer_ind = [0, 4, 8, 12, 16, 20]
        self._finger_indices = (1, 2, 3, 4)

    def process(self, image):
        self._reset_params()
        self.results = self.hands.process(image)
        if self.results.multi_handedness:
            for land, cls in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                self.hands_side[cls.classification[0].label] =\
                    np.array([[xyz.x * self.width, xyz.y * self.height, xyz.z * self.width]
                              for xyz in land.landmark]).astype(np.float32)
                self.visible[cls.classification[0].label] = True

    def palm_center(self, side):
        return np.mean(self.hands_side[side][self.palm_ind, :], axis=0) if self.visible[side] else []

    def index_finger_tip(self, side):
        return self.hands_side[side][8] if self.visible[side] else []

    def is_finger_straight(self, side, finger_index):
        if self.visible[side]:
            _, d_top = self._base_to_tip_dist(side, finger_index)
            _, d_base = self._base_segment_len(side, finger_index)
            return d_top / d_base > self.STRAIGHT_THRESHOLD, self.hands_side[side][4 * (finger_index + 1), :-1]
        else:
            return False, []

    def is_four(self, side):
        four = [self.is_finger_straight(side, f_ind)[0] for f_ind in self._finger_indices]
        if False in four:
            return False, []
        else:
            return True, self.hands_side[side][20, :-1]

    def is_three(self, side):
        three = [self.is_finger_straight(side, f_ind)[0] for f_ind in self._finger_indices[:-1]]
        if False in three:
            return False, []
        elif not self.is_finger_straight(side, 4)[0]:
            return True, self.hands_side[side][16, :-1]
        else:
            return False, []

    def is_two(self, side):
        two = [self.is_finger_straight(side, f_ind)[0] for f_ind in self._finger_indices[:-2]]
        if False in two:
            return False, []
        elif not self.is_finger_straight(side, 4)[0] and not self.is_finger_straight(side, 3)[0]:
            return True, self.hands_side[side][12, :-1]
        else:
            return False, []

    def is_only_finger_straight(self, side, finger_index):
        is_straight, position = self.is_finger_straight(side, finger_index)
        if is_straight:
            others = [self.is_finger_straight(side, f_ind)[0]
                      for f_ind in self._finger_indices if f_ind != finger_index]
            if True in others:
                return False, []
            else:
                return True, position
        else:
            return False, []

    def is_opened_hand(self, side):
        if self.visible[side]:
            base_radius = 0.5 * (
                        np.sqrt(np.sum((self.hands_side[side][5, :] - self.hands_side[side][0, :]) ** 2)) +
                        np.sqrt(np.sum((self.hands_side[side][17, :] - self.hands_side[side][0, :]) ** 2)))
            (xc, yc), outer_radius = cv2.minEnclosingCircle(self.hands_side[side][self.outer_ind, :-1])
            return outer_radius / base_radius > self.OPEN_THRESHOLD, np.array([xc, yc], dtype=np.float32)
        else:
            return False, []

    def _all_fingers_lens_xy(self, side):
        return np.sqrt(np.sum((self.hands_side[side][8::4, :-1] - self.hands_side[side][5::4, :-1]) ** 2, axis=-1))

    def _all_fingers_lens_xyz(self, side):
        return np.sqrt(np.sum((self.hands_side[side][8::4, :] - self.hands_side[side][5::4, :]) ** 2, axis=-1))

    def _base_to_tip_dist(self, side, finger_index):
        base_to_tip = self.hands_side[side][4 * (finger_index + 1), :] - self.hands_side[side][4 * finger_index + 1, :]
        return self._dxy_dr(base_to_tip)

    def _base_segment_len(self, side, finger_index):
        seg_diff = self.hands_side[side][4 * finger_index + 2, :] - self.hands_side[side][4 * finger_index + 1, :]
        return self._dxy_dr(seg_diff)

    @staticmethod
    def _dxy_dr(diff):
        squared = np.power(diff, 2)
        return np.sqrt(squared[0] + squared[1]), np.sqrt(squared[0] + squared[1] + squared[2])

    def _reset_params(self):
        self.hands_side['Right'] = np.array([])
        self.hands_side['Left'] = np.array([])
        self.visible['Right'] = False
        self.visible['Left'] = False

    def close(self):
        self.hands.close()
