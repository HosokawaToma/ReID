import cv2


class ServiceCamera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

    def __del__(self):
        self.camera.release()
        cv2.destroyAllWindows()

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            yield frame
