from dataclasses import dataclass
from data_class.yolo_keypoint import YoloKeypoint

@dataclass
class YoloKeypoints:
    _nose: YoloKeypoint
    _eye_l: YoloKeypoint
    _eye_r: YoloKeypoint
    _ear_l: YoloKeypoint
    _ear_r: YoloKeypoint
    _shoulder_l: YoloKeypoint
    _shoulder_r: YoloKeypoint
    _elbow_l: YoloKeypoint
    _elbow_r: YoloKeypoint
    _wrist_l: YoloKeypoint
    _wrist_r: YoloKeypoint
    _hip_l: YoloKeypoint
    _hip_r: YoloKeypoint
    _knee_l: YoloKeypoint
    _knee_r: YoloKeypoint
    _ankle_l: YoloKeypoint
    _ankle_r: YoloKeypoint

    def __init__(
        self,
        nose: YoloKeypoint,
        eye_l: YoloKeypoint,
        eye_r: YoloKeypoint,
        ear_l: YoloKeypoint,
        ear_r: YoloKeypoint,
        shoulder_l: YoloKeypoint,
        shoulder_r: YoloKeypoint,
        elbow_l: YoloKeypoint,
        elbow_r: YoloKeypoint,
        wrist_l: YoloKeypoint,
        wrist_r: YoloKeypoint,
        hip_l: YoloKeypoint,
        hip_r: YoloKeypoint,
        knee_l: YoloKeypoint,
        knee_r: YoloKeypoint,
        ankle_l: YoloKeypoint,
        ankle_r: YoloKeypoint
    ):
        self._nose = nose
        self._eye_l = eye_l
        self._eye_r = eye_r
        self._ear_l = ear_l
        self._ear_r = ear_r
        self._shoulder_l = shoulder_l
        self._shoulder_r = shoulder_r
        self._elbow_l = elbow_l
        self._elbow_r = elbow_r
        self._wrist_l = wrist_l
        self._wrist_r = wrist_r
        self._hip_l = hip_l
        self._hip_r = hip_r
        self._knee_l = knee_l
        self._knee_r = knee_r
        self._ankle_l = ankle_l
        self._ankle_r = ankle_r

    def get_list(self) -> list[YoloKeypoint]:
        return [
            self._nose,
            self._eye_l,
            self._eye_r,
            self._ear_l,
            self._ear_r,
            self._shoulder_l,
            self._shoulder_r,
            self._elbow_l,
            self._elbow_r,
            self._wrist_l,
            self._wrist_r,
            self._hip_l,
            self._hip_r,
            self._knee_l,
            self._knee_r,
            self._ankle_l,
            self._ankle_r
        ]

    def get_nose(self) -> YoloKeypoint:
        return self._nose

    def get_eye_l(self) -> YoloKeypoint:
        return self._eye_l

    def get_eye_r(self) -> YoloKeypoint:
        return self._eye_r

    def get_ear_l(self) -> YoloKeypoint:
        return self._ear_l

    def get_ear_r(self) -> YoloKeypoint:
        return self._ear_r

    def get_shoulder_l(self) -> YoloKeypoint:
        return self._shoulder_l

    def get_shoulder_r(self) -> YoloKeypoint:
        return self._shoulder_r

    def get_elbow_l(self) -> YoloKeypoint:
        return self._elbow_l

    def get_elbow_r(self) -> YoloKeypoint:
        return self._elbow_r

    def get_wrist_l(self) -> YoloKeypoint:
        return self._wrist_l

    def get_wrist_r(self) -> YoloKeypoint:
        return self._wrist_r

    def get_hip_l(self) -> YoloKeypoint:
        return self._hip_l

    def get_hip_r(self) -> YoloKeypoint:
        return self._hip_r

    def get_knee_l(self) -> YoloKeypoint:
        return self._knee_l

    def get_knee_r(self) -> YoloKeypoint:
        return self._knee_r

    def get_ankle_l(self) -> YoloKeypoint:
        return self._ankle_l

    def get_ankle_r(self) -> YoloKeypoint:
        return self._ankle_r
