# -*-coding:utf-8-*-
import cv2
import numpy as np
import time


class CaptureManager(object):
    def __init__(self, capture, preview_window_manager=None, should_mirror_preview=False):
        self.preview_window_manager = preview_window_manager
        self.should_mirror_preview = should_mirror_preview

        self._capture = capture
        # 给多头摄像机使用
        self._channel = 0
        # 标识是否已经grab()
        self._entered_frame = False
        self._frame = None
        self._image_file_name = None
        self._video_file_name = None
        self._video_encoding = None
        self._video_writer = None
        # 用于计算帧率
        self._start_time = None
        # 已处理过的frame个数
        self._frames_elapsed = 0
        # 估算的帧率，保存视频且无法获取当前帧率时使用
        self._fps_estimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._entered_frame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @frame.setter
    def frame(self, processed_frame):
        if processed_frame is not None:
            self._frame = processed_frame

    @property
    def is_writing_image(self):
        return self._image_file_name is not None

    @property
    def is_writing_video(self):
        return self._video_file_name is not None

    # 调用capture.grab()
    def enter_frame(self):
        assert not self._entered_frame, "上一个enter_frame()没有与之对应的exit_frame()"
        if self._capture is not None:
            self._entered_frame = self._capture.grab()

    # 真正保存帧的地方
    def exit_frame(self):
        if self.frame is None:
            self._entered_frame = False
            return
        # 如果是第一帧，记录开始时间
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        # 否则计算帧率
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1

        # 如果windowManager不为空，则显示帧
        if self.preview_window_manager is not None:
            if self.should_mirror_preview:
                mirrored_frame = np.fliplr(self._frame).copy()
                self.preview_window_manager.show(mirrored_frame)
            else:
                self.preview_window_manager.show(self._frame)

        if self.is_writing_image:
            cv2.imwrite(self._image_file_name, self._frame)
            self._image_file_name = None

        self._write_video_frame()

        self._frame = None
        self._entered_frame = False

    # 记录保存图像的文件名
    def write_image(self, filename):
        self._image_file_name = filename

    # 记录保存视频的文件名和编码信息
    def start_writing_video(self, filename, encoding=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')):
        self._video_file_name = filename
        self._video_encoding = encoding

    # 清除保存视频的信息
    def stop_writing_video(self):
        self._video_file_name = None
        self._video_encoding = None
        self._video_writer = None

    # 写视频帧的地方
    def _write_video_frame(self):
        if not self.is_writing_video:
            return
        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                if self._frames_elapsed < 20:
                    return
                else:
                    fps = self._fps_estimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._video_writer = cv2.VideoWriter(self._video_file_name, self._video_encoding, fps, size)
        self._video_writer.write(self._frame)


class WindowManager(object):
    def __init__(self, window_name, keypress_callback=None):
        self.keypress_callback = keypress_callback
        self._window_name = window_name
        self._is_window_created = False

    @property
    def is_window_created(self):
        return self._is_window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._is_window_created = True

    def show(self, frame):
        cv2.imshow(self._window_name, frame)

    def destroy_window(self):
        cv2.destroyWindow(self._window_name)
        self._is_window_created = False

    def process_events(self):
        keycode = cv2.waitKey(1)
        if self.keypress_callback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypress_callback(keycode)
