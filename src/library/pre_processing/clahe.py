import torch
import kornia


def clahe_gpu(image_tensor: torch.Tensor, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> torch.Tensor:
    """
    PyTorchテンソルに対し、GPUでCLAHEを適用します。(Kornia v0.7.x 対応版)

    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (B, C, H, W), [0.0, 1.0]
        clip_limit (float): コントラスト制限の閾値。
        grid_size (tuple): グリッドサイズ。

    Returns:
        torch.Tensor: CLAHE適用後の3チャンネルRGB画像テンソル。
    """
    if image_tensor.dim() != 4:
        raise ValueError(f"入力テンソルは4階である必要がありますが、{image_tensor.dim()}階です。")

    b, c, h, w = image_tensor.shape

    # --- チャンネル数に応じて処理を分岐 ---

    # グレースケール画像 (C=1)
    if c == 1:
        # kornia.enhance.equalize_clahe 関数を直接適用
        img_eq = kornia.enhance.equalize_clahe(
            image_tensor, clip_limit=clip_limit, grid_size=grid_size)
        return img_eq.repeat(1, 3, 1, 1)

    # RGBA画像 (C=4)
    elif c == 4:
        image_tensor = image_tensor[:, :3, :, :]

    # RGB画像 (C=3)
    lab_img = kornia.color.rgb_to_lab(image_tensor)
    l_channel, a_channel, b_channel = torch.chunk(lab_img, 3, dim=1)
    l_channel_norm = l_channel / 100.0

    # ★★★ 修正点 ★★★
    # kornia.enhance.equalize_clahe 関数を直接呼び出す
    l_channel_eq_norm = kornia.enhance.equalize_clahe(
        l_channel_norm,
        clip_limit=clip_limit,
        grid_size=grid_size
    )

    l_channel_eq = l_channel_eq_norm * 100.0
    lab_img_eq = torch.cat([l_channel_eq, a_channel, b_channel], dim=1)
    rgb_img_eq = kornia.color.lab_to_rgb(lab_img_eq)

    return torch.clamp(rgb_img_eq, 0.0, 1.0)


# --- 実行サンプル ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    dummy_rgb_tensor = torch.rand(4, 3, 256, 256).to(device)
    print(f"\n入力RGBテンソルの形状: {dummy_rgb_tensor.shape}")

    processed_tensor = clahe_gpu(dummy_rgb_tensor)

    print(f"出力RGBテンソルの形状: {processed_tensor.shape}")
    print("GPUでのCLAHE処理が完了しました。")
