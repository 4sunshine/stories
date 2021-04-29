import cv2
import numpy as np
from collections import deque
from PIL import ImageFont, ImageDraw, Image

from media.opencv import init_capture_window, show_image
from tracking.tracking import GoldenTrack
from landmarks.hands import Hands
from media.effects import Warm, WarpingFade, Titles
from media.ffmpeg import OutVideoWriter
from image.opencv import binded_warp_cv_coloured, binded_warp_cv_coloured_image_grid, \
    binded_warp_cv_coloured_centered, binded_warp_cv_coloured_double, binded_warp_cv_displacements
from media.production import VideoToInstagram
from stories.base import Story


def run():
    writer = OutVideoWriter()
    WIDTH = 1280
    HEIGHT = 720
    WINNAME = 'MEDIA'

    cap = init_capture_window(WIDTH, HEIGHT, WINNAME)
    hands = Hands(min_det=0.4, min_track=0.5, width=WIDTH, height=HEIGHT)

    left_index_finger_track = GoldenTrack(history_len=80)
    two_track = GoldenTrack()
    three_track = GoldenTrack()
    four_track = GoldenTrack()

    memorized = False

    effects = deque([Warm()] * 5)
    tracks = deque([left_index_finger_track.get_track_feed, two_track.get_track_feed,
                    three_track.get_track_feed, four_track.get_track_feed])
    gestures = deque([hands.is_only_finger_straight, hands.is_two, hands.is_three, hands.is_four])
    texts = deque(['One', 'Two', 'Three', 'Four'])
    args = deque([('Left', 1), ('Left',), ('Left',), ('Left',)])
    warps_list = [binded_warp_cv_coloured_double, binded_warp_cv_coloured_centered, binded_warp_cv_coloured,
                  binded_warp_cv_coloured_image_grid]
    warps = deque(warps_list)
    fades = deque([WarpingFade(binded_warp_cv_coloured_double)] +
                  [WarpingFade(fn) for fn in warps_list] + [WarpingFade(binded_warp_cv_coloured_image_grid)])
    last_fade = WarpingFade(binded_warp_cv_coloured_double)

    title_content = {'logos': [], 'title': 'FOUR WARPS',
                     'texts': ['+ OpenCV', '+ Mediapipe'], 'target_size': [720, 1280], 'font_size': 96}
    title = Titles(title_content)

    def_effect = Warm()

    ef = effects.pop()
    tr = tracks.pop()
    ges = gestures.pop()
    t = texts.pop()
    a = args.pop()
    w = warps.pop()
    fade = fades.pop()

    print('HERE')

    fontpath = "design/fonts/edo.ttf"
    font = ImageFont.truetype(fontpath, 46)

    last_base_coord = None
    last_dx = None
    try:
        while True:
            success, image = cap.read()

            # FLIP AND RGB OF FRAME
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # PROCESS HANDS
            image.flags.writeable = False

            if ges is not None:
                hands.process(image)
                is_active, pos = ges(*a)
                hist, to_save = tr(is_active, pos)
            else:
                is_active = False

            image.flags.writeable = True

            image = ef.transform(image) if ef is not None else def_effect.transform(image)

            if title is not None:
                image = title.transform(image)
                if is_active and len(hist) > 0:
                    title = None

            if len(hist) > 0 and not to_save:
                x, y = hist[0]
                coord = [int(round(x)), int(round(y))]
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text(coord, t, font=font, fill=(255, 0, 0, 0))
                image = np.array(img_pil)
                if not memorized:
                    fade.renew()
                    initial_point = coord
                    memorized = True
                dx = [coord[0] - initial_point[0], coord[1] - initial_point[1]]

                # if i > 400:
                # THIS PART IS MADE ONLY FOR CAPTURING A SINGLE FRAME WITH DISPLACEMENT
                #     image = binded_warp_cv_displacements(image, initial_point, dx)
                #     cv2.imwrite('displacements.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                #     raise
                image = w(image, initial_point, dx)

            elif not is_active:
                if not last_base_coord is None and not memorized:
                    image = fade.transform(image, last_base_coord.copy(), last_dx.copy()) if fade is not None \
                        else last_fade.transform(image, last_base_coord.copy(), last_dx.copy())

            if to_save:
                if memorized:
                    last_dx = dx
                    last_base_coord = initial_point

                    memorized = False
                ef = effects.pop() if effects else None
                ges = gestures.pop() if gestures else None
                t = texts.pop() if texts else None
                tr = tracks.pop() if tracks else None
                a = args.pop() if args else None
                w = warps.pop() if warps else None
                fade = fades.pop() if fades else None

            # SHOW IMAGE
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            show_image(image, WINNAME)

            writer.write(image)

            if cv2.waitKey(1) & 0xFF == 27:
                writer.close()
                break
    except Exception as e:
        print(e)
        writer.close()


def postprocess():
    vi = VideoToInstagram('out.mp4', 'design/images/red_yellow_orange.png')
    vi.production()


class FourWarps(Story):
    def __init__(self, name='four_warps', run_fn=run, production_fn=postprocess):
        super().__init__(name, run_fn, production_fn)
