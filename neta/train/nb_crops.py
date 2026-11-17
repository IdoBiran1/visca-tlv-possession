"""Training crops of one half: touch_spot.crops at the video's real frame size (117092 and 132831 aren't 4096 x 1080),
on every other frame around each event (every frame on the two other cameras). python nb_crops.py MID H [EVENT_STEP]   (env BOX: YOLO window px)"""
import os, sys
import cv2
sys.path[:0] = [os.getcwd(), os.getcwd() + "/external"]
import touch_spot as T
mid, h = sys.argv[1], int(sys.argv[2])
cap = cv2.VideoCapture(T.video_path(mid, h)); T.W, T.H = int(cap.get(3)), int(cap.get(4)); cap.release()
T.YOLO_OFFS = range(-6, 7, 1 if mid in ("117092", "132831") else 2)   # YOLO finds half as many balls on those two cameras
T.YOLO_BOX = int(os.environ.get("BOX", 320))
T.N_FRAMES = 9                                  # train_ext.py keeps the middle 3, 5 or 9
if len(sys.argv) > 3:
    ev = T.events; T.events = lambda *a: ev(*a)[::int(sys.argv[3])]
T.crops(mid, h)
