"""YOLO管理クラス"""

from dataclasses import dataclass
from typing import List, Tuple
import logging

import numpy as np
from ultralytics import YOLO


@dataclass
class YoloConfig:
    class MODEL:
        model_path: str = "models/yolo11n-pose.pt"
        confidence_threshold: float = 0.5
        person_class_id: int = 0


YOLO_CONFIG = YoloConfig()


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
            self.logger.info(
                f"YOLOモデルを読み込みました: {YOLO_CONFIG.MODEL.model_path}")
        except Exception as e:
            self.logger.error(f"YOLOモデルの初期化に失敗しました: {e}")
            raise Exception(f"YOLOモデルの初期化に失敗しました: {e}")

    def _track_persons(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        フレーム内の人物を検出・追跡する

        :param frame: 入力フレーム
        :return: (バウンディングボックス, 人物画像)のタプルのリスト
        """
        try:
            self.logger.debug(f"YOLO推論開始 - フレームサイズ: {frame.shape}")

            # YOLOで人物検出・追跡を実行
            results = self.model.track(
                frame,
                persist=True,
                classes=[YOLO_CONFIG.MODEL.person_class_id],
                conf=YOLO_CONFIG.MODEL.confidence_threshold,
                verbose=False,
                tracker="bytetrack.yaml"  # より安定した追跡アルゴリズムを使用
            )

            if not results or len(results) == 0:
                self.logger.debug("YOLO推論結果が空です")
                return []

            detections = []
            result = results[0]  # 最初の結果を使用

            self.logger.debug(f"YOLO推論完了 - 結果数: {len(results)}")

            if result.boxes is not None:
                self.logger.debug(f"検出されたボックス数: {len(result.boxes)}")

                for i, box in enumerate(result.boxes):
                    # バウンディングボックス座標を取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())

                    self.logger.debug(
                        f"ボックス {i+1}: 座標=({x1},{y1},{x2},{y2}), 信頼度={confidence:.3f}")

                    # フレーム境界内にクリップ
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))

                    # 人物領域を切り抜き
                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size > 0:
                        bounding_box = np.array([x1, y1, x2, y2])
                        detections.append((bounding_box, person_crop))
                        self.logger.debug(
                            f"人物 {i+1} の切り抜きサイズ: {person_crop.shape}")
                    else:
                        self.logger.warning(f"人物 {i+1} の切り抜きが空です")
            else:
                self.logger.debug("検出されたボックスがありません")

            # デバッグ情報をログ出力
            if detections:
                self.logger.debug(f"人物検出数: {len(detections)}")
            else:
                self.logger.debug("人物が検出されませんでした")

            return detections

        except Exception as e:
            self.logger.error(f"人物追跡中にエラーが発生しました: {e}")
            return []

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
            detections = self._track_persons(frame)
            return detections

        except Exception as e:
            self.logger.error(f"人物切り抜き処理エラー: {e}")
            raise ValueError(f"人物の切り抜きに失敗しました: {e}")
