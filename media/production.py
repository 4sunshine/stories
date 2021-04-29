import cv2
import os
from media.effects import resulting_blur


class VideoToInstagram:
    def __init__(self, video_path, background_path, additional_photos=[], processing_fn=resulting_blur,
                 horizontal_fit=True, fps=25):
        self.video_path = video_path
        self.background_path = background_path
        self.fn = processing_fn
        self.cap = cv2.VideoCapture(self.video_path)
        self.v_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.v_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.h = 1920
        self.w = 1080
        self.horizontal_fit = horizontal_fit
        self.v_target_width, self.v_target_height, self.need_resize = self._check_size()
        _output_basename = f'instagram_{os.path.basename(self.video_path)}'
        self.output_filename = os.path.join(os.path.dirname(self.video_path), _output_basename)
        self.writer = cv2.VideoWriter(self.output_filename,
                                      cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), float(fps), (self.w, self.h))
        self.back = cv2.imread(self.background_path)
        self.additional_photos = [cv2.resize(cv2.imread(path), (self.v_width, self.v_height),
                                             interpolation=cv2.INTER_LANCZOS4) for path in additional_photos]\
            if len(additional_photos) > 0 else []
        self.pause = 10
        assert self.h == self.back.shape[0] and self.w == self.back.shape[1]

    def _check_size(self):
        if self.horizontal_fit and (self.v_width != self.w):
            target_width = self.w
            target_height = int(self.v_height * self.w / self.v_width)
            target_height += target_height % 2
            need_resize = True
        else:
            target_width = self.v_width
            target_height = self.v_height
            need_resize = False
        return target_width, target_height, need_resize

    def production(self):
        start_y_ind = (self.h - self.v_target_height) // 2
        end_y_ind = (self.h - self.v_target_height) // 2 + self.v_target_height
        start_x_ind = (self.w - self.v_target_width) // 2
        end_x_ind = (self.w - self.v_target_width) // 2 + self.v_target_width
        while True:
            ret, frame = self.cap.read()
            if ret:
                if self.need_resize:
                    frame = cv2.resize(frame, (self.v_target_width, self.v_target_height),
                                       interpolation=cv2.INTER_LANCZOS4)
                result = self.back
                result[start_y_ind: end_y_ind, start_x_ind: end_x_ind, :] = frame

                if self.fn is not None:
                    result = self.fn(result, self.v_target_height, self.v_target_width, self.h, self.w)

                self.writer.write(result)
            else:
                for i in range(len(self.additional_photos)):
                    for j in range(self.pause):
                        result = self.back.copy()
                        self.writer.write(result)

                    for _ in range(2 * self.pause):
                        # NEED TO CHECK RESIZE HERE!!!
                        if self.need_resize:
                            self.additional_photos[i] = cv2.resize(self.additional_photos[i],
                                                                   (self.v_target_width, self.v_target_height),
                                                                   interpolation=cv2.INTER_LANCZOS4)
                        result = self.back
                        result[start_y_ind: end_y_ind, start_x_ind: end_x_ind, :] = self.additional_photos[i]

                        if self.fn is not None:
                            result = self.fn(result, self.v_target_height, self.v_target_width, self.h, self.w)

                        self.writer.write(result)

                self.writer.release()
                break
