import cv2
import numpy as np
import os
import moviepy.editor as mpe
import ffmpy
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from modules.parse_poses import parse_poses
from modules.draw import draw_poses


def load_model(model='human-pose-estimation-3d.pth', device='CPU'):
    return InferenceEnginePyTorch(model, device)


def find_humans(frame, net):
    stride = 8
    base_height = 256
    input_scale = base_height / frame.shape[0]
    scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
    scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]

    inference_result = net.infer(scaled_img)
    fx = np.float32(0.8 * frame.shape[1])
    poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, True)
    return poses_2d.copy()


def draw_humans(frame, poses_2d):
    draw_poses(frame, poses_2d)


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Error opening video {}'.format(path))
    return cap


def concat_images(img1, img2, height=500):
    def compress_img(img):
        w = round(img.shape[1] * height / img.shape[0])
        return cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)

    c1 = compress_img(img1)
    c2 = compress_img(img2)
    return np.concatenate((c1, c2), axis=1)


def normalize_pose(p):
    p = p.astype('float64')
    p -= np.repeat(np.expand_dims(p.mean(axis=0), axis=0), p.shape[0], axis=0)
    k = np.median(p[:, 0] ** 2 + p[:, 1] ** 2)
    if k != 0:
        p /= k
    return p


def count_pos_error(pos1, pos2):
    return 0
    p1 = pos1.copy()
    p2 = pos2.copy()

    p1 = normalize_pose(p1)
    p2 = normalize_pose(p2)
    return round(((p1 - p2) ** 2).sum(axis=1).mean().item() * 1000000)


def add_error_on_frame(frame, err):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.0
    thickness = 2
    h, w = frame.shape[: 2]

    def get_text_start_point(center_point, text):
        center_point_x, center_point_y = center_point
        text_sz, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_sz_x, text_sz_y = text_sz
        return (center_point_x - text_sz_x // 2,
                center_point_y + text_sz_y // 2)

    label = str(err)
    x, y = w // 2, h - 30
    cv2.rectangle(frame, (x - 50, y - 20), (x + 50, y + 20), color=[255, 255, 255], thickness=-1)
    cv2.putText(frame, label, get_text_start_point((x, y), label),
                font, thickness=thickness, color=[0, 0, 0], fontScale=font_scale)

    return frame


def print_grade(total_err):
    print()
    print('Total err: {}'.format(total_err))

    grades = [(75, 5), (150, 4), (225, 3), (300, 2), (np.inf, 1)]

    for bound, grade in grades:
        if total_err <= bound:
            print('Grade: {}'.format(grade))
            return total_err, grade


def modify_two_videos(cap1, cap2, frame_modifier, out=None, logger=None):
    fps = round(cap1.get(cv2.CAP_PROP_FPS))
    cap1.set(cv2.CAP_PROP_FPS, fps)
    cap2.set(cv2.CAP_PROP_FPS, fps)
    frames = round(min(
        cap1.get(cv2.CAP_PROP_FRAME_COUNT),
        cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    i = 0
    while cap1.isOpened() and cap2.isOpened():
        print('\rFrame {}/{}'.format(i, frames), end='')
        if logger is not None:
            logger.log(i, frames)
        i += 1
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame = frame_modifier(frame1, frame2)

        if out is not None:
            out.write(frame)

    cap1.release()
    cap2.release()
    if out is not None:
        out.release()


def smooth_poses(prv, cur, nxt):
    e1 = count_pos_error(prv, cur)
    e2 = count_pos_error(cur, nxt)
    e3 = count_pos_error(prv, nxt)
    res = cur
    if e1 + e2 > 1.5 * e3:
        res = (prv + nxt) // 2
    return res


class Logger:
    def __init__(self, callback, l_threshold, r_threshold):
        self.callback = callback
        self.l_threshold = l_threshold
        self.r_threshold = r_threshold

    def log(self, x, y):
        """x out of y done"""
        if self.callback is not None:
            self.callback((self.l_threshold * (y - x) + self.r_threshold * x) / y)


def draw_arrows(frame, train_pose, my_pose):
    return frame
    train_normalized = normalize_pose(train_pose.copy())
    my_normalized = normalize_pose(my_pose.copy())
    train_center = train_pose.mean(axis=0)
    my_center = my_pose.mean(axis=0)
    for i in range(my_normalized.shape[0]):
        d = my_normalized[i] - train_normalized[i]
        dist = (d[0] ** 2).sum()
        if dist > 0.0005:
            x = tuple(np.rint(my_pose[i]).astype(int))
            y = tuple(np.rint(train_pose[i] - train_center + my_center).astype(int))
            frame = cv2.arrowedLine(frame, x, y,
                                    color=[0, 255, 0], thickness=3)
    return frame


def make_video(path1, path2, out_path, res_estimator, processing_log=None):
    prv1, cur1 = None, None
    prv2, cur2 = None, None
    prv_frame1, prv_frame2 = None, None

    cap1 = open_video(path1)
    cap2 = open_video(path2)
    fps = round(cap1.get(cv2.CAP_PROP_FPS))

    errors = []
    h1, w1 = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)), round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2, w2 = round(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), round(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h, w = concat_images(np.zeros((h1, w1, 3)), np.zeros((h2, w2, 3))).shape[: 2]

    def frame_modifier(frame1, frame2):
        nonlocal prv1, prv2, cur1, cur2, prv_frame1, prv_frame2, h, w
        nxt1 = find_humans(frame1, res_estimator)
        nxt2 = find_humans(frame2, res_estimator)
        if prv1 is None:
            prv1, prv2 = nxt1, nxt2
        elif cur1 is None:
            cur1, cur2 = nxt1, nxt2
        else:
            cur1 = smooth_poses(prv1, cur1, nxt1)
            cur2 = smooth_poses(prv2, cur2, nxt2)
        if prv_frame1 is not None:
            draw_humans(prv_frame1, cur1)
            draw_humans(prv_frame2, cur2)
            frame2 = draw_arrows(frame2, cur1.copy(), cur2.copy())
            frame = concat_images(prv_frame1, prv_frame2)
        else:
            frame = concat_images(frame1, frame2)

        prv_frame1, prv_frame2 = frame1, frame2
        prv1, prv2, cur1, cur2 = cur1, cur2, nxt1, nxt2

        err = 0
        if cur1 is not None:
            err = count_pos_error(cur1, cur2)
            errors.append(err)

        return add_error_on_frame(frame, err)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    c = out_path.split('.')
    tmp_path = '.'.join(c[:-1]) + '_tmp.' + c[-1]
    out_cap = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    modify_two_videos(cap1, cap2, frame_modifier, out_cap, Logger(processing_log, 0, 100))

    # set audio
    output_video = mpe.VideoFileClip(tmp_path)
    audio_background = mpe.VideoFileClip(path1).audio.subclip(t_end=output_video.duration)
    final_video = output_video.set_audio(audio_background)
    final_video.write_videofile(out_path)
    os.remove(tmp_path)

    if len(errors) != 0:
        total = round(np.mean(errors).item())
        return print_grade(total)


def convert_video(video_path, out_path):
    flags = '-r 24 -codec copy'
    ff = ffmpy.FFmpeg(inputs={video_path: None}, outputs={out_path: flags})
    ff.run()


if __name__ == '__main__':
    path1 = '../../../kek1.mp4'
    path2 = '../../../kek2.mp4'
    out_path = '../../../kek5.mp4'

    e = load_model()

    make_video(path1, path2, out_path, e)