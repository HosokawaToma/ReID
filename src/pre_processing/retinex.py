import torch
import kornia

# 以前修正したSSR関数 (これは変更なし)


def ssr_gpu(image_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    device = image_tensor.device
    epsilon = 1e-6
    log_i = torch.log1p(image_tensor.clamp(min=epsilon))
    kernel_size = int(sigma * 4) * 2 + 1
    gaussian_blur = kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size), (sigma, sigma)).to(device)
    log_l = gaussian_blur(log_i)
    log_r = log_i - log_l
    return log_r

# MSR関数 (これも変更なし)


def msr_gpu(image_tensor: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    ssr_results = [ssr_gpu(image_tensor, sigma) for sigma in sigmas]
    msr_result = torch.stack(ssr_results, dim=0).mean(dim=0)
    return msr_result

# ★★★ 新しいMSRCR関数 ★★★


def msrcr_gpu(image_tensor: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """
    色回復付きマルチスケールレティネックス (MSRCR) をGPUで計算します。
    入力は必ずRGB形式のテンソルである必要があります。
    """
    # まず、MSRを計算します
    msr_log_r = msr_gpu(image_tensor, sigmas)

    # --- ここからが色回復処理 ---
    epsilon = 1e-6
    # 元画像の各チャンネルの合計を計算
    intensity = image_tensor.sum(dim=1, keepdim=True)

    # 各色チャンネルの割合を計算
    # 元画像 I_k / (Σ I_i) を対数領域で計算
    color_ratio = torch.log1p(
        (image_tensor / intensity.clamp(min=epsilon)).clamp(min=epsilon))

    # MSRの結果に、色割合を反映させる
    msrcr_log_r = msr_log_r * color_ratio

    # 結果を[0, 1]の範囲に正規化して返す
    # 各チャンネルごとに最小値・最大値で正規化
    min_val = msrcr_log_r.min(
        dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = msrcr_log_r.max(
        dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    range_val = (max_val - min_val).clamp(min=epsilon)

    return (msrcr_log_r - min_val) / range_val


# --- 実行フロー ---
if __name__ == '__main__':
    import cv2
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 画像読み込み (BGR)
    image_bgr_np = cv2.imread("your_image.jpg")

    # 2. ★最重要★ BGR -> RGBに変換
    image_rgb_np = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)

    # 3. テンソルに変換 (RGBのまま)
    tensor_rgb = torch.from_numpy(image_rgb_np.astype(np.float32)) \
        .permute(2, 0, 1) / 255.0
    tensor_rgb = tensor_rgb.unsqueeze(0).to(device)

    # 4. MSRCRを適用
    sigmas = [15, 80, 200]
    processed_tensor = msrcr_gpu(tensor_rgb, sigmas)

    # 5. 結果をNumpyに戻す
    output_np_rgb = (processed_tensor.squeeze(0).cpu().permute(
        1, 2, 0).numpy() * 255).astype(np.uint8)

    # 6. ★最重要★ 保存時にRGB -> BGRに戻す
    output_np_bgr = cv2.cvtColor(output_np_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("msrcr_result.jpg", output_np_bgr)

    print("MSRCR処理が完了しました。")
