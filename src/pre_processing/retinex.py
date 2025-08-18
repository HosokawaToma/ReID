import torch
import kornia

# --- 修正版: Single-Scale Retinex (SSR) のGPU実装 ---


def ssr_gpu(image_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    単一スケールのRetinexをGPUで計算します。(カーネルサイズ計算を修正)

    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (B, C, H, W), [0.0, 1.0]
        sigma (float): ガウシアンフィルタの標準偏差（ぼかしの強さ）

    Returns:
        torch.Tensor: SSR処理後のテンソル（対数領域）
    """
    device = image_tensor.device
    epsilon = 1e-6
    log_i = torch.log1p(image_tensor.clamp(min=epsilon))

    # ★★★ ここからが修正点 ★★★
    # sigmaからカーネルサイズを計算 (必ず正の奇数になるように)
    # 一般的に、カーネルサイズはsigmaの約4倍の2倍+1が良いとされます
    kernel_size = int(sigma * 4) * 2 + 1

    # 計算したカーネルサイズを明示的に指定
    gaussian_blur = kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size), (sigma, sigma)
    ).to(device)
    # ★★★ ここまでが修正点 ★★★

    log_l = gaussian_blur(log_i)
    log_r = log_i - log_l

    return log_r
