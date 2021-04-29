import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline
from PIL import ImageFont, ImageDraw, Image


class BaseEffect:
    def __init__(self, content, target_size=(720, 1280), initial_coord=(0, 0)):
        self.t_h, self.t_w = target_size
        self.y0, self.x0 = initial_coord
        self.ticks = 0
        self.content = self._init_content(content)

    def _init_content(self, content):
        return None

    def show(self, *args, **kwargs):
        return self.content

    def transform(self, image, *args, **kwargs):
        return image

    def renew(self):
        pass


class MovingImage(BaseEffect):
    def __init__(self, filename, target_size=(720, 1280), initial_coord=(0, 0)):
        super().__init__(filename, target_size, initial_coord)
        self.content = self._init_content(filename)
        self.vx = 4
        self.vy = 2

    def _init_content(self, content):
        return cv2.imread(content, cv2.COLOR_BGR2RGB)

    def show(self):
        self.x0 += self.vx
        self.y0 += self.vy
        self.ticks += 1
        return self.content[self.y0: self.y0 + self.t_h, self.x0: self.x0 + self.t_w, :]

    def transform(self, image, *args, **kwargs):
        return cv2.addWeighted(self.show(), 0.4, image, 0.6, 0)


def spreadLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


class Warm(BaseEffect):
    def __init__(self, content=None):
        super().__init__(content)

    def _init_content(self, content):
        self.increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        self.decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])

    def transform(self, image, *args, **kwargs):
        red_channel, green_channel, blue_channel = cv2.split(image)
        red_channel = cv2.LUT(red_channel, self.increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, self.decreaseLookupTable).astype(np.uint8)
        return cv2.merge((red_channel, green_channel, blue_channel))


class Cold(BaseEffect):
    def __init__(self, content=None):
        super().__init__(content)

    def _init_content(self, content):
        self.increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        self.decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])

    def transform(self, image, *args, **kwargs):
        red_channel, green_channel, blue_channel = cv2.split(image)
        red_channel = cv2.LUT(red_channel, self.decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, self.increaseLookupTable).astype(np.uint8)
        return cv2.merge((red_channel, green_channel, blue_channel))


class WarpingFade(BaseEffect):
    def __init__(self, content):
        super().__init__(content)

    def _init_content(self, content):
        self._ticks = 0
        self._fn = content

    def transform(self, image, *args, **kwargs):
        argument = self._ticks * (1. + self._ticks / 50.) / 5.
        exponent = - self._ticks / 35
        base_point, base_dx = args
        base_dx[0] = base_dx[0] * np.cos(argument) * (2.71 ** exponent)
        base_dx[1] = base_dx[1] * np.cos(argument) * (2.71 ** exponent)
        self._ticks += 1
        return self._fn(image, base_point, base_dx)

    def renew(self):
        self._ticks = 0


class Titles(BaseEffect):
    def __init__(self, content=None):
        if content is None:
            content = {'logos': [], 'title': 'Title', 'texts': [], 'target_size': [720, 1280], 'font_size': 64}
        super().__init__(content)

    def _init_content(self, content):
        self._ticks = 0
        self._logos = content['logos']
        self._title = content['title']
        self._texts = content['texts']
        self._h, self._w = content['target_size']
        fontpath = "design/fonts/HeyNovember-eZZ2l.ttf"
        self._font_size_title = content['font_size']
        self._font_size_text = int(self._font_size_title * 0.7)
        self._font_title = ImageFont.truetype(fontpath, self._font_size_title)
        self._font = ImageFont.truetype(fontpath, self._font_size_text)
        self._n_elements = len(self._logos) + len(self._title) + len(self._texts)
        self._ticks_per_text = 10
        self._fills = [(51, 153, 255, 0), (0, 204, 0, 0)]

    def transform(self, image, *args, **kwargs):
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, self._h // 2 - self._font_size_title), self._title, font=self._font_title, fill=(255, 51, 0, 0))
        self._ticks += 1
        for i in range(min(len(self._texts), self._ticks // self._ticks_per_text)):
            draw.text((10, self._h // 2 + i * int(self._font_size_text * 1.2)), self._texts[i],
                      font=self._font, fill=self._fills[i])
        return np.array(img_pil)

    def renew(self):
        self._ticks = 0


class ChannelRemove(BaseEffect):
    def __init__(self, zero_index=1):
        super().__init__(content=zero_index)

    def _init_content(self, content):
        return content

    def transform(self, image, *args, **kwargs):
        image[:, :, self.content] = np.zeros_like(image[:, :, self.content], dtype=np.uint8)
        return image


class ChannelShifting(BaseEffect):
    def __init__(self, shifting=4):
        super().__init__(content=shifting)

    def _init_content(self, content):
        return content

    def transform(self, image, *args, **kwargs):
        image[:, self.content:, 0] = image[:, :-self.content, 0]
        image[:, :-self.content, 2] = image[:, self.content:, 2]
        return image


def edges_smoothing(im1, im2, n_pixels):
    assert im1.shape == im2.shape, 'Shape mismatch'
    im1_mean = np.mean(im1, axis=0, keepdims=True)
    im2_mean = np.mean(im2, axis=0, keepdims=True)
    linsp = np.linspace(0, 1, n_pixels, dtype=np.float32)[:, np.newaxis, np.newaxis]
    d_image = im2_mean - im1_mean
    result = np.round(im1_mean + d_image * linsp).astype(np.uint8)
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(result, -1, kernel)


def resulting_blur(result, v_h, v_w, h, w, n_border=120):
    start_ind, end_ind = (h - v_h) // 2, (h - v_h) // 2 + v_h
    last_line_video = result[end_ind - 2: end_ind, :, :]
    last_line_result = result[end_ind + n_border: end_ind + n_border + 2, :, :]
    result[end_ind: end_ind + n_border, :, :] = edges_smoothing(last_line_video, last_line_result, n_border)
    first_line_video = result[start_ind: start_ind + 2, :, :]
    first_line_result = result[start_ind - n_border - 2: start_ind - n_border, :, :]
    result[start_ind - n_border: start_ind, :, :] = edges_smoothing(first_line_result, first_line_video, n_border)
    return result

