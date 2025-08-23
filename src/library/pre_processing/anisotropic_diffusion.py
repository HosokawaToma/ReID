import torch


def anisotropic_diffusion(image_tensor: torch.Tensor, n_iter: int, delta_t: float, kappa: float) -> torch.Tensor:
    """
    異方性拡散フィルタリングを適用する関数
    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (H, W)
        n_iter (int): 反復回数
        delta_t (float): 時間ステップ
        kappa (float): 拡散を制御する係数
    Returns:
        torch.Tensor: 処理済み画像テンソル
    """
    device = image_tensor.device

    # 差分演算用の畳み込みカーネルを定義
    k_n = torch.tensor([[0., 1., 0.], [0., -1., 0.], [0., 0., 0.]],
                       dtype=u.dtype, device=device).view(1, 1, 3, 3)
    k_s = torch.tensor([[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]],
                       dtype=u.dtype, device=device).view(1, 1, 3, 3)
    k_e = torch.tensor([[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]],
                       dtype=u.dtype, device=device).view(1, 1, 3, 3)
    k_w = torch.tensor([[0., 0., 0.], [1., -1., 0.], [0., 0., 0.]],
                       dtype=u.dtype, device=device).view(1, 1, 3, 3)

    for _ in range(n_iter):
        # 境界を拡張
        u_padded = torch.nn.functional.pad(u.unsqueeze(
            0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')

        # 各方向への輝度勾配（差分）を計算
        nabla_n = torch.conv2d(u_padded, k_n, padding=0)
        nabla_s = torch.conv2d(u_padded, k_s, padding=0)
        nabla_e = torch.conv2d(u_padded, k_e, padding=0)
        nabla_w = torch.conv2d(u_padded, k_w, padding=0)

        # 拡散係数 g(nabla_u) を計算
        g_n = 1.0 / (1.0 + (nabla_n / kappa)**2)
        g_s = 1.0 / (1.0 + (nabla_s / kappa)**2)
        g_e = 1.0 / (1.0 + (nabla_e / kappa)**2)
        g_w = 1.0 / (1.0 + (nabla_w / kappa)**2)

        # 方程式を更新
        u += delta_t * (g_n * nabla_n + g_s * nabla_s +
                        g_e * nabla_e + g_w * nabla_w)

    return u.squeeze()
