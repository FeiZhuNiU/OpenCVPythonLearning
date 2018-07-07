# -*-coding:utf-8-*-
import cv2

from cvgui.managers import WindowManager, CaptureManager


class Cameo(object):
    def __init__(self, video):
        self._window_manager = WindowManager('Cameo', self.on_keypress)
        self._capture_manager = CaptureManager(cv2.VideoCapture(video), self._window_manager, True)

    def run(self):
        self._window_manager.create_window()
        while self._window_manager.is_window_created:
            self._capture_manager.enter_frame()
            frame = self._capture_manager.frame

            # TODO 对frame进行处理

            self._capture_manager.exit_frame()
            self._window_manager.process_events()

    def on_keypress(self, keycode):
        """
        space -> 截屏
        tab -> 开始/结束录屏
        escape -> 退出
        """
        if keycode == 32:  # space
            self._capture_manager.write_image('screenshot.png')
        elif keycode == 9:  # tab
            if not self._capture_manager.is_writing_video:
                self._capture_manager.start_writing_video('screencast.avi')
            else:
                self._capture_manager.stop_writing_video()
        elif keycode == 27:  # esc
            self._window_manager.destroy_window()


if __name__ == "__main__":
    Cameo("street.mp4").run()
