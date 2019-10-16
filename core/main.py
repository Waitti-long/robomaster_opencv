import cv2
import core.utils as utils
import joblib
import numpy as np
import ctypes
import time


now_num = 1
so = ctypes.cdll.LoadLibrary
lib = so("../interfaces/lib_send.so")
clf = joblib.load("../model/mlfbig.m")
video = cv2.VideoCapture(0)
video.set(3, 1920)
video.set(4, 1080)


def send_message(hor, ver):
    global lib
    lib.send_message(hor, ver)


def is_run(now_location, run_error=10):
    center_loc = [0, 0]
    error_all = 0
    for _i in range(2):
        error_all += np.abs(now_location[i] - center_loc[i])
    return run_error > error_all


def get_num_list(frame):
    num_set = set()
    num_loc_list = []
    squares = utils.findSquare(frame)
    for square in squares:
        hog = utils.getHog(square[0])
        sc = clf.predict(np.array([hog]))
        if sc != 0:
            if sc not in num_set:
                num_set.add(sc)
                x, y, w, h = square[1][0], square[1][1], square[1][2], square[1][3]
                horizontal, vertical = utils.pnp(np.float64([[x, y], [x, x + w], [x + w, y + h], [x, y + h]]))
                li = (sc, (horizontal, vertical))
                num_loc_list.append(li)
            else:
                return None
    if len(num_loc_list) > 0:
        return num_loc_list
    else:
        return None


def catch_frame():
    global video
    success, frame = video.read()
    while not success:
        success, frame = video.read()
    return frame


def get_num():
    frame = catch_frame()
    num_list = get_num_list(frame)
    while num_list is None:
        frame = catch_frame()
        num_list = get_num_list(frame)
    return num_list


def start(wait_seconds=0.1, every_seconds=2.1):
    nums = get_num()
    for num in nums:
        if num[0] == now_num:
            send_message(num[1][0], num[1][1])
            time.sleep(wait_seconds)
            nums_2 = get_num()
            for num_2 in nums_2:
                if num_2[0] == now_num:
                    if not is_run(num_2[1][0], num_2[1][0]):
                        time.sleep(every_seconds)
                        return "still_ok"
                    else:
                        return "is_run"
    return "not_ok"


def control_run(relax=0.01, every_seconds=2.1):
    time_start = time.time()
    while True:
        nums = get_num()
        for num in nums:
            if num[0] == now_num:
                send_message(num[1][0], num[1][1])
                break
        time.sleep(relax)
        time_now = time.time()
        if time_now - time_start >= every_seconds:
            break


if __name__ == '__main__':
    time_start = time.time()
    while True:
        res = start()
        if res == "still_ok":
            now_num += 1
            if now_num >= 6:
                break
        elif res == "not_ok":
            res = start()
        elif res == "is_run":
            now_num +=1
            control_run()
    time_end = time.time()
    print("time_count: ", time_end - time_start)
