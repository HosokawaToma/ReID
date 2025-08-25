import cv2
import numpy as np


def apply_reinhard_tonemapping(image, gamma=1.0, intensity=1.0):
    """
    Reinhardトーンマッピングを適用して画像を補正する。

    Args:
        image (np.array): 処理対象の画像（OpenCVのBGR形式）。
        gamma (float): 明るさのガンマ調整。
        intensity (float): トーンマッピングの強度。

    Returns:
        np.array: 補正後の画像。
    """
    # 画像をfloat32型に変換（Reinhardトーンマッピングの入力要件）
    image = image.astype(np.float32) / 255.0

    # Reinhardトーンマッピングオブジェクトを作成
    # デフォルトのパラメーターで、シンプルかつ安定した結果が得られる
    tonemapper = cv2.createTonemapReinhard(
        gamma=gamma,
        intensity=intensity,
        light_adapt=0.0,  # 通常は0でOK
        color_adapt=0.0   # 通常は0でOK
    )

    # トーンマッピングを適用
    hdr_image = tonemapper.process(image)

    # 0-1の範囲にクリップし、0-255に戻す
    hdr_image = np.clip(hdr_image * 255, 0, 255).astype(np.uint8)

    return hdr_image
