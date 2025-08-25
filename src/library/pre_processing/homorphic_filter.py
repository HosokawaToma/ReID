import torch
import torch.nn.functional as F
import numpy as np
import cv2


def homomorphic_filter(image_tensor: torch.Tensor, d0: float = 30.0, gamma_l: float = 0.5, gamma_h: float = 2.0, c: float = 1.0) -> torch.Tensor:
    """
    HSV色空間で輝度成分のみにホモモルフィックフィルタリングを適用する関数

    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (0-1に正規化、形状: [B, C, H, W] または [C, H, W])
        d0 (float): カットオフ周波数
        gamma_l (float): 低周波成分のゲイン（照明成分の寄与度）
        gamma_h (float): 高周波成分のゲイン（反射成分の寄与度）
        c (float): フィルタの鋭さ
    Returns:
        torch.Tensor: フィルタ適用後の画像テンソル
    """
    is_3d = False
    if image_tensor.dim() == 3:
        is_3d = True
        image_tensor = image_tensor.unsqueeze(0)

    device = image_tensor.device

    # --- 1. RGBからHSVへ変換し、Vチャンネルを抽出 ---
    hsv_tensor = _rgb_to_hsv(image_tensor)
    v_channel = hsv_tensor[:, 2:3, :, :]

    # --- 2. Vチャンネルにホモモルフィックフィルタを適用 ---
    # 対数変換の前にスケーリング
    log_v = torch.log(v_channel * 255 + 1)

    # フーリエ変換
    fft_v = torch.fft.fft2(log_v)
    fft_shifted = torch.fft.fftshift(fft_v)

    # ガウシアン・ハイパスフィルタの作成
    rows, cols = v_channel.shape[-2:]
    center_row, center_col = rows // 2, cols // 2
    j = torch.arange(cols, device=device) - center_col
    i = torch.arange(rows, device=device) - center_row
    Y, X = torch.meshgrid(i, j, indexing='ij')
    dist_sq = X**2 + Y**2
    filter_kernel = (gamma_h - gamma_l) * \
        (1 - torch.exp(-c * dist_sq / (d0**2))) + gamma_l
    filter_kernel = filter_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # フィルタ適用
    filtered_fft = fft_shifted * filter_kernel

    # 逆フーリエ変換
    ifft_shifted = torch.fft.ifftshift(filtered_fft)
    ifft_v = torch.fft.ifft2(ifft_shifted)
    filtered_log_v = torch.real(ifft_v)

    # --- 3. 指数変換とクリッピング ---
    processed_v = torch.exp(filtered_log_v) - 1

    # Vチャンネルは0-255の範囲に収める
    processed_v = torch.clamp(processed_v, 0, 255) / 255.0

    # --- 4. 処理済みVチャンネルを元のHSVテンソルに戻し、RGBへ変換 ---
    hsv_tensor[:, 2:3, :, :] = processed_v
    result_tensor = _hsv_to_rgb(hsv_tensor)

    if is_3d:
        result_tensor = result_tensor.squeeze(0)

    return result_tensor

# --- RGB <-> HSV 変換ヘルパー関数 ---


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb.split(1, dim=1)
    max_c, max_idx = torch.max(rgb, dim=1, keepdim=True)
    min_c = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = max_c - min_c

    hue = torch.zeros_like(max_c)

    # ブロードキャストを考慮してtorch.whereを使用
    hue_r = (60 * (((g - b) / (delta + 1e-6)) % 6) + 360) % 360
    hue_g = (60 * (((b - r) / (delta + 1e-6)) + 2) + 360) % 360
    hue_b = (60 * (((r - g) / (delta + 1e-6)) + 4) + 360) % 360

    hue = torch.where(max_c == r, hue_r, hue)
    hue = torch.where(max_c == g, hue_g, hue)
    hue = torch.where(max_c == b, hue_b, hue)

    saturation = torch.where(
        max_c != 0, delta / (max_c + 1e-6), torch.zeros_like(max_c))
    value = max_c

    return torch.cat([hue / 360, saturation, value], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv.split(1, dim=1)

    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_int = i.int()

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    # torch.whereを使用して、i_intの値に応じてr, g, bを計算
    r = torch.where(i_int == 0, v, torch.where(i_int == 1, q, torch.where(
        i_int == 2, p, torch.where(i_int == 3, p, torch.where(i_int == 4, t, v)))))
    g = torch.where(i_int == 0, t, torch.where(i_int == 1, v, torch.where(
        i_int == 2, v, torch.where(i_int == 3, q, torch.where(i_int == 4, p, p)))))
    b = torch.where(i_int == 0, p, torch.where(i_int == 1, p, torch.where(
        i_int == 2, t, torch.where(i_int == 3, v, torch.where(i_int == 4, v, q)))))

    return torch.cat([r, g, b], dim=1)
