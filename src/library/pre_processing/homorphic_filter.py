import torch

def homomorphic_filter(image_tensor: torch.Tensor, d0: float = 30.0, gamma_l: float = 0.5, gamma_h: float = 2.0, c: float = 1.0) -> torch.Tensor:
    """
    ホモモルフィックフィルタリングを適用する関数

    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (0-1に正規化)
        d0 (float): カットオフ周波数
        gamma_l (float): 低周波成分のゲイン（照明成分の寄与度）
        gamma_h (float): 高周波成分のゲイン（反射成分の寄与度）
        c (float): フィルタの鋭さ
    Returns:
        torch.Tensor: フィルタ適用後の画像テンソル
    """
    # --- 1. 対数変換 ---
    # log(1+I) を計算。小さな値を加えてlog(0)を避ける
    log_image = torch.log1p(image_tensor)

    # --- 2. フーリエ変換 ---
    fft_image = torch.fft.fft2(log_image)
    fft_shifted = torch.fft.fftshift(fft_image)

    # --- 3. ハイパスフィルタの作成 ---
    rows, cols = image_tensor.shape
    center_row, center_col = rows // 2, cols // 2

    # 画像中心からの距離を計算するためのグリッドをGPU上に作成
    j = torch.arange(cols, device=image_tensor.device) - center_col
    i = torch.arange(rows, device=image_tensor.device) - center_row
    Y, X = torch.meshgrid(i, j, indexing='ij')

    # 中心からの距離の2乗を計算
    dist_sq = X**2 + Y**2

    # ガウシアン・ハイパスフィルタの計算
    # H(u,v) = (γH - γL) * (1 - exp(-c * D(u,v)^2 / D0^2)) + γL
    filter_kernel = (gamma_h - gamma_l) * \
        (1 - torch.exp(-c * dist_sq / (d0**2))) + gamma_l

    # --- 4. フィルタ適用 ---
    filtered_fft = fft_shifted * filter_kernel

    # --- 5. 逆フーリエ変換 ---
    ifft_shifted = torch.fft.ifftshift(filtered_fft)
    ifft_image = torch.fft.ifft2(ifft_shifted)

    # 実数部を取り出す
    filtered_log_image = torch.real(ifft_image)

    # --- 6. 指数変換 ---
    # 対数変換を元に戻す (expm1は exp(x)-1 を高精度に計算する)
    result_image = torch.expm1(filtered_log_image)

    # --- 表示用に0-1の範囲に正規化 ---
    result_image = (result_image - result_image.min()) / \
        (result_image.max() - result_image.min())

    return result_image
