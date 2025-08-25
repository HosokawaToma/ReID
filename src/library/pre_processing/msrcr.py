import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2


def msrcr_enhance(
    image_tensor: torch.Tensor,
    scales: list = [15, 80, 250],
    gain: float = 2.0,
    offset: float = 128.0,
    alpha: float = 125.0,
    beta: float = 46.0
) -> torch.Tensor:
    """
    MSRCR (Multi-Scale Retinex with Color Restoration) を用いた画像強調。
    安定性と高い補正能力を両立した、実績のある手法です。

    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (C, H, W) or (B, C, H, W), 値は[0, 1]の範囲。
        scales (list): Retinexのスケール（ガウシアンブラーのシグマ値）。
        gain (float): 最終的なコントラスト調整のゲイン。
        offset (float): 最終的なコントラスト調整のオフセット。
        alpha (float): 色復元の強度。
        beta (float): 色復元のゲイン。
    Returns:
        torch.Tensor: 処理済みの画像テンソル。
    """
    # --- 入力テンソルの形状を (B, C, H, W) に統一 ---
    is_3d = False
    if image_tensor.dim() == 3:
        is_3d = True
        image_tensor = image_tensor.unsqueeze(0)

    device = image_tensor.device
    epsilon = 1e-6

    # --- 1. マルチスケールRetinex (MSR) ---
    # 対数空間で処理
    log_image = torch.log(image_tensor + epsilon)

    msr_output = torch.zeros_like(log_image)
    for sigma in scales:
        # torchvisionの関数で高速なガウシアンブラーを適用
        log_blurred = torch.log(TF.gaussian_blur(
            image_tensor, kernel_size=int(sigma*4)+1, sigma=sigma) + epsilon)
        msr_output += (log_image - log_blurred)

    msr_output /= len(scales)  # 各スケールの結果を平均

    # --- 2. 色復元 (Color Restoration) ---
    # 元画像のRGBチャンネル間の比率（色情報）を計算
    rgb_sum = torch.sum(image_tensor, dim=1, keepdim=True)
    color_restoration_factor = torch.log(
        alpha * image_tensor + epsilon) - torch.log(rgb_sum + epsilon)
    color_restoration_factor *= beta

    # MSRの結果に色情報を掛け合わせる
    msrcr_output = msr_output * color_restoration_factor

    # --- 3. 最終的なコントラスト調整 ---
    # MSRCRの結果を0-255のスケールに変換
    min_val = msrcr_output.min(
        dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = msrcr_output.max(
        dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    msrcr_output = (msrcr_output - min_val) / \
        (max_val - min_val + epsilon) * 255.0

    # ゲインとオフセットを適用して最終調整
    result = gain * msrcr_output - offset

    # [0, 255]の範囲にクリップし、[0, 1]の範囲に戻す
    result = torch.clamp(result, 0, 255) / 255.0

    # --- 元の形状に戻して返す ---
    if is_3d:
        result = result.squeeze(0)

    return result


# --- 実行サンプル ---
if __name__ == '__main__':
    try:
        # ★★★ ご自身の暗い画像、照明ムラのある画像のパスに変更してください ★★★
        image_path = 'path/to/your/dark_or_uneven_image.jpg'
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(
            image_rgb).permute(2, 0, 1).float() / 255.0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        image_tensor = image_tensor.to(device)

        # MSRCRによる補正を実行
        corrected_tensor = msrcr_enhance(image_tensor)

        # 結果を保存できる形式に変換
        corrected_image_np = corrected_tensor.cpu().permute(1, 2, 0).numpy()
        corrected_image_bgr = cv2.cvtColor(
            (corrected_image_np * 255).astype(np.uint8), cv2.COLOR_RGB_BGR)

        output_path = 'corrected_image_msrcr.jpg'
        cv2.imwrite(output_path, corrected_image_bgr)
        print(f"補正された画像を '{output_path}' に保存しました。")

        # オリジナルと結果を並べて比較画像を保存
        display_image = np.hstack([image_bgr, corrected_image_bgr])
        cv2.imwrite('comparison_msrcr.jpg', display_image)
        print("比較画像 'comparison_msrcr.jpg' を保存しました。")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
