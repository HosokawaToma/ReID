"""シンプルなマルチカメラRe-IDアプリケーション"""
import argparse
import logging
import signal
import sys
from typing import List

import cv2
import numpy as np

from managers.yolo_manager import YoloManager
from managers.reid_manager import ReIDManager
from managers.post_processing_manager import KReciprocalManager
from config import APP_CONFIG, COLOR_PALETTE, REID_CONFIG
from utilities.person_tracker import PersonTracker

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCameraReIDApp:
    """シンプルなマルチカメラRe-IDアプリケーション"""

    def __init__(self, reid_backend: str):
        self.reid_backend = reid_backend
        self.cameras = {}
        self.running = False

        # シグナルハンドラーを設定
        signal.signal(signal.SIGINT, self._signal_handler)

        # コンポーネントの初期化
        logger.info("コンポーネントを初期化しています...")
        self.yolo_manager = YoloManager()
        self.reid_manager = ReIDManager(self.reid_backend)
        self.person_tracker = PersonTracker(
            backend=self.reid_backend, reid_config=REID_CONFIG
        )

        # K-Reciprocal Re-ranking管理の初期化
        if REID_CONFIG.use_re_ranking:
            self.k_reciprocal_manager = KReciprocalManager(
                k1=REID_CONFIG.rerank_k1,
                k2=REID_CONFIG.rerank_k2,
                lambda_value=REID_CONFIG.rerank_lambda
            )
            logger.info("K-Reciprocal Re-ranking管理を初期化しました")
        else:
            self.k_reciprocal_manager = None

        logger.info("初期化完了")

    def _signal_handler(self, sig, frame):
        """シグナルハンドラー"""
        logger.info("終了シグナルを受信しました")
        self.running = False
        self._cleanup()
        sys.exit(0)

    def discover_cameras(self, max_cameras: int = 10) -> List[int]:
        """利用可能なカメラを検出する"""
        available_cameras = []

        for camera_id in range(max_cameras):
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(camera_id)
                        logger.info(f"カメラ {camera_id} が利用可能です")
                    cap.release()
            except Exception as e:
                logger.debug(f"カメラ {camera_id} の検出に失敗: {e}")

        return available_cameras

    def draw_person_info(self, frame: np.ndarray, person_id: int, bbox: np.ndarray) -> np.ndarray:
        """フレームに人物情報を描画する"""
        if frame is None or bbox is None:
            return frame

        frame_copy = frame.copy()

        try:
            x1, y1, x2, y2 = map(int, bbox)

            # 人物IDに基づいて色を選択
            color_idx = person_id % len(COLOR_PALETTE.colors)
            color = COLOR_PALETTE.colors[color_idx]

            # バウンディングボックスを描画
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            # ラベルテキスト
            label = f"ID: {person_id}"

            # ラベル背景を描画
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            # ラベルテキストを描画
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception as e:
            logger.error(f"人物情報描画エラー: {e}")
            return frame

        return frame_copy

    def process_frame(self, frame: np.ndarray, camera_id: int) -> np.ndarray:
        """フレーム処理"""
        try:
            processed_frame = frame.copy()

            # YOLO推論で人物検出
            person_info_list = self.yolo_manager.track_and_detect_persons(
                frame)

            if not person_info_list:
                return processed_frame

            # 検出された各人物を処理
            for bounding_box, person_crop in person_info_list:
                try:
                    # ReID特徴抽出
                    features = self.reid_manager.extract_features(
                        person_crop, camera_id)

                    # 人物ID割り当て
                    person_id = self.person_tracker.assign_person_id(features, self.k_reciprocal_manager)

                    # 描画
                    processed_frame = self.draw_person_info(
                        processed_frame, person_id, bounding_box
                    )

                    logger.debug(f"カメラ{camera_id}: 人物ID {person_id} を検出")

                except Exception as e:
                    logger.error(f"人物処理エラー (カメラ{camera_id}): {e}")
                    continue

            return processed_frame

        except Exception as e:
            logger.error(f"フレーム処理エラー (カメラ{camera_id}): {e}")
            return frame

    def run(self) -> None:
        """アプリケーションのメインループ"""
        try:
            # カメラを検出
            camera_ids = self.discover_cameras()
            if not camera_ids:
                logger.error("利用可能なカメラが見つかりませんでした")
                return

            logger.info(f"利用可能なカメラ: {camera_ids}")

            # カメラを開く
            for camera_id in camera_ids:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, APP_CONFIG.camera_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                            APP_CONFIG.camera_height)
                    cap.set(cv2.CAP_PROP_FPS, APP_CONFIG.camera_fps)
                    self.cameras[camera_id] = cap
                    logger.info(f"カメラ {camera_id} を開始しました")

            if not self.cameras:
                logger.error("カメラを開けませんでした")
                return

            self.running = True
            logger.info("メインループを開始します。'q'キーで終了します。")

            # メインループ
            while self.running:
                frames_to_display = {}

                # 各カメラからフレームを取得・処理
                for camera_id, cap in self.cameras.items():
                    ret, frame = cap.read()
                    if ret:
                        processed_frame = self.process_frame(frame, camera_id)
                        frames_to_display[camera_id] = processed_frame

                # フレームを表示
                for i, (camera_id, frame) in enumerate(frames_to_display.items()):
                    window_name = f"Camera {camera_id}"
                    cv2.imshow(window_name, frame)

                    # ウィンドウの位置を調整（重ならないように）
                    cv2.moveWindow(window_name, i * 650, 0)

                    logger.debug(f"フレームを表示: {window_name}")

                # キー入力チェック
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("終了キーが押されました")
                    break

        except KeyboardInterrupt:
            logger.info("ユーザーによる中断")
        except Exception as e:
            logger.error(f"アプリケーション実行エラー: {e}")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """終了処理"""
        logger.info("アプリケーションを停止しています...")

        self.running = False

        # カメラを解放
        for camera_id, cap in self.cameras.items():
            cap.release()
            logger.info(f"カメラ {camera_id} を停止しました")

        # ウィンドウを閉じる
        cv2.destroyAllWindows()

        logger.info("アプリケーションを終了しました")


def main(reid_backend: str):
    """アプリケーションのメイン関数"""
    app = SimpleCameraReIDApp(reid_backend)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="シンプルなマルチカメラ対応 リアルタイム人物再識別システム")
    parser.add_argument(
        "--reid_backend", type=str, default="clip",
        choices=["clip", "trans_reid", "la_transformer"],
        help="使用するReIDバックエンド"
    )
    args = parser.parse_args()
    main(args.reid_backend)
