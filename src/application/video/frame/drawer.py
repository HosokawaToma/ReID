import cv2
import numpy as np
from typing import Tuple


class ApplicationVideoFrameDrawer:
    def draw(self, frame: np.ndarray, person_id: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        color = self.get_color(person_id)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        frame = cv2.putText(frame, f"{person_id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        return frame

    def get_color(self, person_id: int) -> Tuple[int, int, int]:
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
