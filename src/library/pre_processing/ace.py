import torch


def ace_filter(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    ACEの核心部分をPyTorchで実装
    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (0-1の正規化済、C, H, W)
    Returns:
        torch.Tensor: 処理済み画像テンソル
    """
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    image_reshaped = image_tensor.view(3, -1)  # (3, H*W)

    # ピクセル間の距離を計算（高速化のため行列演算）
    dist_matrix_x = image_reshaped.unsqueeze(2)  # (3, N, 1)
    dist_matrix_y = image_reshaped.unsqueeze(1)  # (3, 1, N)

    # 3チャンネル分の差分行列を計算
    # 差分行列 (3, N, N)
    diff_matrix = dist_matrix_x - dist_matrix_y

    # Sign関数を適用: -1, 0, 1
    sign_matrix = torch.sign(diff_matrix)

    # 全てのピクセルペアに対する差分を足し合わせる
    # (3, N)
    sum_diff = torch.sum(sign_matrix, dim=2)

    # 正規化
    normalized_sum = sum_diff / (H * W)

    # 元の画像に加算して調整
    ace_tensor = image_reshaped + normalized_sum

    # 0-1にクランプし、元の形状に戻す
    ace_tensor = torch.clamp(ace_tensor, 0, 1).view(3, H, W)

    return ace_tensor
