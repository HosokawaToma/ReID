"""YOLO管理クラス"""

from typing import List, Tuple, Optional
import logging

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import YOLO_CONFIG
from exceptions import ModelLoadError, ModelInferenceError

logger = logging.getLogger(__name__)


def initialize_yolo_model() -> YOLO:
    """
    YOLOモデルを初期化する

    :return: 初期化されたYOLOモデル
    :raises Exception: モデルの初期化に失敗した場合
    """
    try:
        model = YOLO(YOLO_CONFIG.model_path)
        logger.info(f"YOLOモデルを読み込みました: {YOLO_CONFIG.model_path}")
        return model
    except Exception as e:
        logger.error(f"YOLOモデルの初期化に失敗しました: {e}")
        raise


def track_persons(model: YOLO, frame: np.ndarray) -> List[Results]:
    """
    フレーム内の人物を検出・追跡する

    :param model: YOLOモデル
    :param frame: 入力フレーム
    :return: YOLO結果のリスト
    """
    try:
        # YOLOで人物検出・追跡を実行
        results = model.track(
            frame,
            persist=True,
            classes=[YOLO_CONFIG.person_class_id],
            conf=YOLO_CONFIG.confidence_threshold,
            verbose=False
        )
        return results
    except Exception as e:
        logger.error(f"人物追跡中にエラーが発生しました: {e}")
        from exceptions import ModelInferenceError
        raise ModelInferenceError(f"YOLO推論中にエラーが発生しました: {e}")


def validate_detection_results(results: List[Results]) -> bool:
    """
    検出結果が有効かどうかを検証する

    :param results: YOLO結果のリスト
    :return: 有効な検出結果がある場合True
    """
    if not results or len(results) == 0:
        return False

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return False

    return True


def extract_person_crop_from_box(box, frame: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    バウンディングボックスから人物の切り抜き画像を抽出する

    :param box: YOLOのバウンディングボックス
    :param frame: 元フレーム
    :return: (track_id, bounding_box, person_crop) のタプル
    :raises ValueError: 無効なボックスまたはフレームの場合
    """
    if frame is None or frame.size == 0:
        raise ValueError("無効なフレームです")

    try:
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

        bounding_box = np.array([x1, y1, x2, y2])

        return bounding_box, person_crop

    except Exception as e:
        logger.error(f"人物切り抜き処理エラー: {e}")
        raise ValueError(f"人物の切り抜きに失敗しました: {e}")


class YoloManager:
    """YOLOモデルの管理と人物検出・追跡を行うクラス"""

    def __init__(self):
        """YoloManagerを初期化"""
        self.model = None
        self._is_initialized = False
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        YOLOモデルを初期化する

        :raises ModelLoadError: モデルの初期化に失敗した場合
        """
        try:
            logger.info("YOLOモデルの初期化を開始します")
            self.model = initialize_yolo_model()
            self._is_initialized = True
            logger.info("YOLOモデルの初期化が完了しました")
        except Exception as e:
            self._is_initialized = False
            logger.error(f"YOLOモデルの初期化に失敗しました: {e}")
            raise ModelLoadError(f"YOLOモデルの初期化に失敗しました: {e}")

    def track_and_detect_persons(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """
        入力画像からYOLOの人物推論を行い、バウンディングボックスを作成する

        :param frame: 入力フレーム (numpy配列)
        :return: 人物情報リスト: [(track_id, bounding_box, person_crop), ...]
        :raises ModelInferenceError: 推論中にエラーが発生した場合
        :raises RuntimeError: モデルが初期化されていない場合
        """
        if not self._is_initialized or self.model is None:
            raise RuntimeError(
                "YOLOモデルが初期化されていません。initialize_model()を先に呼び出してください")

        if frame is None or frame.size == 0:
            logger.warning("無効なフレームが提供されました")
            return []

        try:
            # YOLOで人物を検出・追跡
            results = track_persons(self.model, frame)

            if not validate_detection_results(results):
                logger.debug("人物が検出されませんでした")
                return []

            # 検出された人物の情報を抽出
            person_info_list = []
            result = results[0]

            for box in result.boxes:
                try:
                    bounding_box, person_crop = extract_person_crop_from_box(
                        box, frame)
                    person_info_list.append(
                        (bounding_box, person_crop))
                    logger.debug(f"人物を検出しました")
                except ValueError as e:
                    logger.warning(f"人物の切り抜きに失敗しました: {e}")
                    continue

            logger.debug(f"合計 {len(person_info_list)} 人の人物を検出しました")
            return person_info_list

        except Exception as e:
            logger.error(f"人物検出・追跡中にエラーが発生しました: {e}")
            raise ModelInferenceError(f"人物検出・追跡中にエラーが発生しました: {e}")

    def is_initialized(self) -> bool:
        """
        モデルが初期化されているかどうかを確認する

        :return: 初期化済みの場合True、そうでなければFalse
        """
        return self._is_initialized and self.model is not None

    def get_model_info(self) -> dict:
        """
        モデル情報を取得する

        :return: モデル情報の辞書
        """
        if not self._is_initialized or self.model is None:
            return {
                "initialized": False,
                "model_path": YOLO_CONFIG.model_path,
                "confidence_threshold": YOLO_CONFIG.confidence_threshold
            }

        return {
            "initialized": True,
            "model_path": YOLO_CONFIG.model_path,
            "confidence_threshold": YOLO_CONFIG.confidence_threshold,
            "person_class_id": YOLO_CONFIG.person_class_id
        }
