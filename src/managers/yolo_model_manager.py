"""YOLO管理クラス"""

from typing import List
import logging

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import YOLO_CONFIG

class YoloModelManager():
    """YOLOモデルの管理と人物検出・追跡を行うクラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_yolo_model()

    def _initialize_yolo_model(self) -> None:
        """
        YOLOモデルを初期化する

        :raises Exception: モデルの初期化に失敗した場合
        """
        try:
            self.model = YOLO(YOLO_CONFIG.MODEL.model_path)
            self.logger.info(f"YOLOモデルを読み込みました: {YOLO_CONFIG.MODEL.model_path}")
        except Exception as e:
            self.logger.error(f"YOLOモデルの初期化に失敗しました: {e}")
            raise Exception(f"YOLOモデルの初期化に失敗しました: {e}")

    def _track_persons(self, frame: np.ndarray) -> List[Results]:
        """
        フレーム内の人物を検出・追跡する

        :param frame: 入力フレーム
        :return: YOLO結果のリスト
        """
        try:
            # YOLOで人物検出・追跡を実行
            return self.model.track(
                frame,
                persist=True,
                classes=[YOLO_CONFIG.MODEL.person_class_id],
                conf=YOLO_CONFIG.MODEL.confidence_threshold,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"人物追跡中にエラーが発生しました: {e}")
            raise Exception(f"人物追跡中にエラーが発生しました: {e}")

    def extract_person_crop_from_box(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        フレームから人物の切り抜き画像を抽出する

        :param frame: 元フレーム
        :return: 人物の切り抜き画像のリスト
        :raises ValueError: 無効なボックスまたはフレームの場合
        """
        if frame is None or frame.size == 0:
            raise ValueError("無効なフレームです")

        try:
            results = self._track_persons(frame)

            if len(results) == 0:
                self.logger.debug("人物が検出されませんでした")
                return None

            bounding_boxes = []

            for result in results:
                for box in result.boxes:
                    # バウンディングボックス座標を取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # フレーム境界内にクリップ
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))

                    # 人物領域を切り抜き
                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size == 0:
                        raise ValueError("切り抜き画像が空です")

                    bounding_boxes.append(np.array([x1, y1, x2, y2]))

            return bounding_boxes
        except Exception as e:
            self.logger.error(f"人物切り抜き処理エラー: {e}")
            raise ValueError(f"人物の切り抜きに失敗しました: {e}")
