import cv2
import numpy as np
import os
from media.effects import resulting_blur


class VideoToInstagram:
    def __init__(self, video_path, background_path, additional_photos=[], processing_fn=resulting_blur):
        self.video_path = video_path
        self.background_path = background_path
        self.fn = processing_fn
        self.cap = cv2.VideoCapture(self.video_path)
        self.v_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.v_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.h = 1920
        self.w = 1080
        _output_basename = f'instagram_{os.path.basename(self.video_path)}'
        self.output_filename = os.path.join(os.path.dirname(self.video_path), _output_basename)
        self.writer = cv2.VideoWriter(self.output_filename,
                                      cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 25., (self.w, self.h))
        self.back = cv2.imread(self.background_path)
        self.additional_photos = [cv2.resize(cv2.imread(path), (self.v_width, self.v_height),
                                             interpolation=cv2.INTER_LANCZOS4) for path in additional_photos]\
            if len(additional_photos) > 0 else []
        self.pause = 10
        assert self.h == self.back.shape[0] and self.w == self.back.shape[1]

    def production(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                result = self.back
                result[(self.h - self.v_height) // 2: (self.h - self.v_height) // 2 + self.v_height,
                (self.w - self.v_width) // 2: (self.w - self.v_width) // 2 + self.v_width, :] = frame
                if self.fn is not None:
                    result = self.fn(result, self.v_height, self.v_width, self.h, self.w)
                self.writer.write(result)
            else:
                for i in range(len(self.additional_photos)):
                    for j in range(self.pause):
                        result = self.back.copy()
                        self.writer.write(result)

                    for _ in range(2 * self.pause):
                        result = self.back
                        result[(self.h - self.v_height) // 2: (self.h - self.v_height) // 2 + self.v_height,
                        (self.w - self.v_width) // 2: (self.w - self.v_width) // 2 + self.v_width, :] =\
                            self.additional_photos[i]
                        if self.fn is not None:
                            result = self.fn(result, self.v_height, self.v_width, self.h, self.w)
                        self.writer.write(result)

                self.writer.release()
                break
