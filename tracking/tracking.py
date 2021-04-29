import numpy as np


class Track:
    MAX_EMPTY = 10
    EPSILON_MULT = 0.05

    def __init__(self, history_len=40, vector_dim=2, conf_memory_len=3, conf_initial_state=False,
                 approx=True):
        self.history_len = history_len
        self.vector_dim = vector_dim
        self.current_len = 0
        self.confidence_trace = BooleanTrack(conf_memory_len, conf_initial_state)

        self.history = np.zeros((history_len, vector_dim), dtype=np.float32)
        self.predicted = np.zeros(vector_dim, dtype=np.float32)
        self.empty = 0

        self.approx = approx

    def get_dx(self):
        if self.current_len > 4:
            return (2 * (self.history[-2] - self.history[-4]) + self.history[-1] - self.history[-5]) / 8
        elif self.current_len in (3, 4):
            return (self.history[-1] - self.history[-3]) / 2
        else:
            return np.zeros(self.vector_dim)

    def get_track(self):
        return self.history[-1: self.history_len - self.current_len: -1, :]

    # def _smooth_curve(self):
    #     epsilon = 0.02 * cv2.arcLength(self.get_track(), False)
    #     approx = cv2.approxPolyDP(self.get_track(), epsilon, False)
    #
    #     self.current_len = len(approx)
    #     print(self.current_len)
    #     print(np.shape(approx))
    #     self.history[-1: self.history_len - self.current_len: -1, :] = np.squeeze(approx, axis=1)
    #     #return uniform_filter1d(self.history[-1: self.history_len - self.current_len: -1, :], 3, axis=0)[0]

    def add_to_memory(self, data):
        if self.current_len < 3:
            if len(data) > 0:
                self._append(data)
            else:
                self._append(self.history[-1, :])
        else:
            prediction = self._predict()
            if len(data) > 0:
                self._append((data + prediction) / 2)
                self.empty = 0
            else:
                self._append(prediction)
                self.empty += 1

        # if self.approx and self.current_len > 3:
        #     self._smooth_curve()
            # self.smooth_curve()
            # if self.empty >= self.MAX_EMPTY:
            #     return self.release()
            # else:
            #     return self.get_track()

    def release(self):
        self.history = np.zeros((self.history_len, self.vector_dim), dtype=np.float32)
        self.current_len = 0
        self.empty = 0

    def _append(self, value):
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1, :] = value
        self.current_len += 1
        self.current_len = min(self.current_len, self.history_len)

    def _predict(self):
        return self.history[-1] + self._speed() + 0.5 * self._acceleration()

    def _speed(self):
        return self.history[-1] - self.history[-2]

    def _acceleration(self):
        return self.history[-1] - 2 * self.history[-2] + self.history[-3]


class BooleanTrack:
    def __init__(self, memory_len=3, initial_state=False):
        self.history = [initial_state] * memory_len
        self.memory_len = memory_len
        self.initial_state = initial_state
        self.fire = False

    def check_fire(self):
        if not self.fire and self.is_confident():
            self.fire = True
        elif self.fire and self.need_release():
            self.fire = False
        return self.fire

    def append(self, value):
        self.history[:-1] = self.history[1:]
        self.history[-1] = value

    def is_confident(self):
        is_confident = True
        for h in self.history:
            is_confident &= h
        return is_confident

    def need_release(self):
        need_remove = False
        for h in self.history:
            need_remove |= h
        return not need_remove

    def refresh(self):
        self.history = [self.initial_state] * self.memory_len


class GoldenTrack:
    def __init__(self, history_len=40, vector_dim=2, conf_memory_len=3, conf_initial_state=False):
        self.val_trace = Track(history_len, vector_dim)
        self.conf_trace = BooleanTrack(conf_memory_len, conf_initial_state)
        self._was_fire = False

    def _feed(self, confidence, value):
        self.conf_trace.append(confidence)
        # RETURNS FIRE STATUS AND NEED TO SAVE BOOLEAN VARIABLE
        if self.conf_trace.check_fire():
            self._was_fire = True
            self.val_trace.add_to_memory(value)
            return True, False
        elif self._was_fire:
            self._was_fire = False
            return False, True
        else:
            return False, False

    def get_track_feed(self, confidence, value):
        # WRAPPER AROUND ._FEED METHOD
        fire, need_to_store = self._feed(confidence, value)
        if fire:
            return self.val_trace.get_track(), False
        elif need_to_store:
            history = self.val_trace.get_track()
            self.val_trace.release()
            return history, True
        else:
            return [], False

    def get_dx_feed(self, confidence, value):
        fire, need_to_store = self._feed(confidence, value)
        if fire:
            return self.val_trace.get_dx(), False
        elif need_to_store:
            history = self.val_trace.get_dx()
            self.val_trace.release()
            return history, True
        else:
            return [], False
