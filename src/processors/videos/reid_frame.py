import numpy as np
from typing import Callable, List, Tuple
from data_class.person_detections import PersonDetections
import random
from data_class.person_data_set_features import PersonDataSetFeatures
import torch
import cv2

class VideosReIDFrameProcessor:
    def __init__(
        self,
        extract_person_detections: Callable[[np.ndarray], List[PersonDetections]],
        extract_features: Callable[[np.ndarray], np.ndarray],
        assign_person_id: Callable[[np.ndarray, np.ndarray, np.ndarray], int]
    ):
        self.extract_person_detections = extract_person_detections
        self.extract_features = extract_features
        self.assign_person_id = assign_person_id
        self.gallery_features = PersonDataSetFeatures(
            persons_id=[], cameras_id=[], views_id=[], features=torch.tensor([]))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        person_detections = self.extract_person_detections(frame)
        for person_detection in person_detections:
            feat = self.extract_features(person_detection.person_crop)
            person_id = self.assign_person_id(feat)
            frame = self._draw_detection(frame, person_detection.bounding_box, person_id)
        return frame

    def _draw_detection(self, frame: np.ndarray, bounding_box: np.ndarray, person_id: int) -> np.ndarray:
        color = self._get_color_for_id(person_id)
        frame = cv2.rectangle(
            frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color, 2)
        frame = cv2.putText(
            frame, str(person_id), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _get_color_for_id(self, person_id: int) -> Tuple[int, int, int]:
        colors = [
            (0, 0, 139),      # 濃い赤
            (139, 0, 0),      # 濃い青
            (0, 100, 0),      # ダークグリーン
            (128, 0, 128),    # 紫
            (0, 128, 128),    # 濃いシアン
            (128, 128, 0),    # オリーブ
            (255, 69, 0),     # オレンジレッド
            (72, 61, 139),    # ダークブルー
            (0, 0, 128),      # ネイビーブルー
            (85, 107, 47),    # ダークオリーブグリーン
            (139, 69, 19),    # サドルブラウン
            (0, 139, 139),    # ダークシアン
            (46, 139, 87),    # シーグリーン
            (160, 32, 240),   # パープル
            (0, 191, 255),    # ディープスカイブルー
            (255, 140, 0),    # ダークオレンジ
            (0, 128, 0),      # グリーン
            (0, 0, 205),      # ミディアムブルー
            (34, 139, 34),    # フォレストグリーン
            (255, 20, 147),   # ディープピンク
            (25, 25, 112),    # ミッドナイトブルー
            (128, 0, 0),      # マルーン
            (0, 255, 127),    # スプリンググリーン
            (255, 0, 127),    # ローズ
            (70, 130, 180),   # スチールブルー
            (0, 206, 209),    # ダークターコイズ
            (199, 21, 133),   # ミディアムバイオレットレッド
            (255, 0, 255),    # マゼンタ
            (0, 191, 255),    # ディープスカイブルー
            (139, 0, 139),    # ダークマゼンタ
        ]

        return colors[person_id % len(colors)]
