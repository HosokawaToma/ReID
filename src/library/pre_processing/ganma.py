import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # ガンマ値の逆数を計算
    invGamma = 1.0 / gamma

    # 0から255までのピクセル値に対応するガンマ補正後の値を計算し、テーブルを作成
    # 値を[0, 1]に正規化してからガンマ補正を適用し、[0, 255]に戻す
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # ルックアップテーブル（LUT）を使って画像にガンマ補正を適用
    return cv2.LUT(image, table)

# --- ここからが実行部分 ---


# 画像を読み込む
image_path = 'your_image.jpg'  # ここに画像のパスを指定してください
original = cv2.imread(image_path)

if original is None:
    print(f"エラー: 画像ファイル '{image_path}' を読み込めません。")
else:
    # ガンマ値を調整して暗い部分を明るくする
    # gamma < 1.0 で明るくなる
    gamma_bright = 0.5
    adjusted_bright = adjust_gamma(original, gamma=gamma_bright)

    # ガンマ値を調整して全体を暗くする（参考）
    # gamma > 1.0 で暗くなる
    gamma_dark = 1.5
    adjusted_dark = adjust_gamma(original, gamma=gamma_dark)

    # 結果を表示
    cv2.imshow("Original", original)
    cv2.imshow(f"Gamma Corrected (gamma={gamma_bright})", adjusted_bright)
    cv2.imshow(f"Gamma Corrected (gamma={gamma_dark})", adjusted_dark)

    # キーが押されるまで待機
    cv2.waitKey(0)
    cv2.destroyAllWindows()
