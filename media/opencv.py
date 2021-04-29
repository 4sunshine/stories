import cv2


def init_capture_window(width=1280, height=720, winname='MEDIA'):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cv2.namedWindow(winname)
    return cap


def show_image(image, winname, x=600, y=0):
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname, image)