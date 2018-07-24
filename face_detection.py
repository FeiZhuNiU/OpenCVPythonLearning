import cv2


def generate():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    # read()方法读取视频下一帧到frame，当读取不到内容时返回false!
    count = 0
    while success and cv2.waitKey(1) & 0xFF != ord('q'):
        # 等待1毫秒读取键键盘输入，最后一个字节是键盘的ASCII码。ord()返回字母的ASCII码
        cv2.imshow('frame', frame)
        success, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if faces is not None:
            for (x, y, w, h) in faces:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

                cv2.imwrite('./data/eric/%s.pgm' % str(count), f)
                count += 1
    cv2.destroyAllWindows()
    camera.release()


if __name__ == "__main__":
    generate()
