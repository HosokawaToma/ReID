import torch


def logarithmic_transform(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    対数変換を適用する関数
    I_out = c * log(1 + I_in)
    """
    # log(1 + I) を計算 (log1pは log(1+x) を高精度に計算する)
    log_transformed = torch.log1p(image_tensor)

    # 表示用に0-1の範囲に正規化
    log_transformed = (log_transformed - log_transformed.min()) / \
        (log_transformed.max() - log_transformed.min())
    return log_transformed
