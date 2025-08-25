import torch
import torch.nn.functional as F


def ace_filter(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    メモリ効率的なACEフィルタの実装
    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (0-1の正規化済、C, H, W)
    Returns:
        torch.Tensor: 処理済み画像テンソル
    """
    # 入力テンソルの形状を確認
    if image_tensor.dim() == 4:
        # [B, C, H, W] の場合
        batch_size, num_channels, height, width = image_tensor.shape
        image_tensor = image_tensor.view(-1, num_channels, height, width)
        result = _apply_ace_filter_2d(image_tensor)
        result = result.view(batch_size, num_channels, height, width)
    elif image_tensor.dim() == 3:
        # [C, H, W] の場合
        result = _apply_ace_filter_2d(image_tensor.unsqueeze(0)).squeeze(0)
    else:
        raise ValueError(f"予期しないテンソル次元: {image_tensor.dim()}")

    return result


def _apply_ace_filter_2d(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    2次元テンソルにACEフィルタを適用する内部関数
    """
    batch_size, num_channels, height, width = image_tensor.shape

    # 局所的なACEフィルタを適用（カーネルサイズ7x7）
    kernel_size = 7
    padding = kernel_size // 2

    # 各チャンネルに対して局所的な統計を計算
    result = torch.zeros_like(image_tensor)

    for b in range(batch_size):
        for c in range(num_channels):
            # 現在のチャンネルの画像
            current_channel = image_tensor[b, c:c+1, :, :]

            # 局所的な平均と標準偏差を計算
            local_mean = F.avg_pool2d(
                current_channel, kernel_size, stride=1, padding=padding)
            local_var = F.avg_pool2d(
                current_channel**2, kernel_size, stride=1, padding=padding) - local_mean**2
            local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))

            # 局所的な正規化
            normalized = (current_channel - local_mean) / (local_std + 1e-8)

            # グローバルな統計も考慮
            global_mean = torch.mean(current_channel)
            global_std = torch.std(current_channel)

            # 局所とグローバルの統計を組み合わせ
            enhanced = normalized * global_std + global_mean

            # 0-1の範囲にクランプ
            enhanced = torch.clamp(enhanced, 0, 1)

            result[b, c:c+1, :, :] = enhanced

    return result
