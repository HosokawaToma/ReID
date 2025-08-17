import torch
import kornia


def clahe_gpu(image_tensor: torch.Tensor, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> torch.Tensor:
    """
    PyTorchテンソルに対し、GPUでCLAHEを適用します。
    元のOpenCVコードのロジック（L*a*b*色空間での輝度チャンネル処理）を再現します。

    Args:
        image_tensor (torch.Tensor): 入力画像テンソル。
            - 形状: (B, C, H, W)
            - データ型: torch.float
            - 値の範囲: [0.0, 1.0]
            - 色チャンネル: 1 (Grayscale), 3 (RGB), or 4 (RGBA)
        clip_limit (float): コントラスト制限の閾値。
        grid_size (tuple): グリッドサイズ。

    Returns:
        torch.Tensor: CLAHE適用後の3チャンネルRGB画像テンソル。
    """
    # 入力テンソルが4階テンソル (B, C, H, W) であることを確認
    if image_tensor.dim() != 4:
        raise ValueError(f"入力テンソルは4階である必要がありますが、{image_tensor.dim()}階です。")

    b, c, h, w = image_tensor.shape
    device = image_tensor.device

    # CLAHEモジュールを初期化
    clahe = kornia.enhance.CLAHE(
        clip_limit=clip_limit, grid_size=grid_size).to(device)

    # --- チャンネル数に応じて処理を分岐 ---

    # 1. グレースケール画像 (C=1)
    if c == 1:
        # そのままCLAHEを適用
        img_eq = clahe(image_tensor)
        # 3チャンネルに拡張して返す (OpenCVの挙動を再現)
        return img_eq.repeat(1, 3, 1, 1)

    # 2. RGBA画像 (C=4)
    elif c == 4:
        # Alphaチャンネルを破棄してRGBにする
        image_tensor = image_tensor[:, :3, :, :]

    # 3. RGB画像 (C=3)
    # L*a*b*色空間に変換して処理
    # KorniaはRGBを想定 (OpenCVのBGRとは異なるので注意)
    lab_img = kornia.color.rgb_to_lab(image_tensor)

    # L, a, bチャンネルに分離
    l_channel, a_channel, b_channel = torch.chunk(lab_img, 3, dim=1)

    # Lチャンネルの値の範囲は [0, 100] なので、CLAHEが想定する [0, 1] に正規化
    l_channel_norm = l_channel / 100.0

    # LチャンネルにのみCLAHEを適用
    l_channel_eq_norm = clahe(l_channel_norm)

    # 再び [0, 100] の範囲に戻す
    l_channel_eq = l_channel_eq_norm * 100.0

    # チャンネルを再結合
    lab_img_eq = torch.cat([l_channel_eq, a_channel, b_channel], dim=1)

    # RGB色空間に戻す
    rgb_img_eq = kornia.color.lab_to_rgb(lab_img_eq)

    # 値が[0.0, 1.0]の範囲に収まるようにclip
    return torch.clamp(rgb_img_eq, 0.0, 1.0)


# --- 実行サンプル ---
if __name__ == '__main__':
    # GPUが利用可能か確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # (Batch, Channel, Height, Width) のダミーRGB画像テンソルを作成
    # 値の範囲は [0.0, 1.0]
    dummy_rgb_tensor = torch.rand(4, 3, 256, 256).to(device)
    print(f"\n入力RGBテンソルの形状: {dummy_rgb_tensor.shape}")

    # GPUでCLAHEを適用
    processed_tensor = clahe_gpu(dummy_rgb_tensor)

    print(f"出力RGBテンソルの形状: {processed_tensor.shape}")
    print("GPUでのCLAHE処理が完了しました。")

    # グレースケール画像のテスト
    dummy_gray_tensor = torch.rand(4, 1, 256, 256).to(device)
    print(f"\n入力Grayテンソルの形状: {dummy_gray_tensor.shape}")
    processed_gray_tensor = clahe_gpu(dummy_gray_tensor)
    print(f"出力Gray→RGBテンソルの形状: {processed_gray_tensor.shape}")
