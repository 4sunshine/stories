import cv2
import subprocess
import numpy as np


class WebCam:
    def __init__(self, width=1280, height=720, fps=25):
        ffmpg_cmd = [
            'ffmpeg',
            # FFMPEG_BIN,
            '-f', 'video4linux2',
            '-input_format', 'mjpeg',
            '-framerate', f'{fps}',
            '-video_size', f'{width}x{height}',
            '-i', '/dev/video0',
            '-pix_fmt', 'bgr24',  # opencv requires bgr24 pixel format
            '-an', '-sn',  # disable audio processing
            '-vcodec', 'rawvideo',
            '-f', 'image2pipe',
            '-',  # output to go to stdout
        ]
        self.h = height
        self.w = width
        self.read_amount = height * width * 3
        self.process = subprocess.Popen(ffmpg_cmd, stdout=subprocess.PIPE, bufsize=10**8)

    def get_frame(self, flip=True, rgb=True):
        # transform the bytes read into a numpy array
        frame = np.frombuffer(self.process.stdout.read(self.read_amount), dtype=np.uint8)
        frame = frame.reshape((self.h, self.w, 3))  # height, width, channels
        self.process.stdout.flush()
        if flip and rgb:
            return cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        elif flip and not rgb:
            return cv2.flip(frame, 1)
        else:
            return frame

    def close(self):
        self.process.terminate()


class OutVideoWriter:
    def __init__(self, output_filename='out.mp4', width=1280, height=720, fps=25):
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', f'{width}x{height}',
                   '-pix_fmt', 'bgr24',
                   '-r', f'{fps}',
                   '-i', '-',
                   '-an',
                   '-vcodec', 'libx264',
                   '-preset', 'ultrafast',
                   # '-tune', 'film',
                   '-qp', '0',
                   output_filename]

        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def close(self):
        self.process.stdin.close()
        self.process.wait()
        print(self.process.poll())