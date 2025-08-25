import torch
from pytorch_wavelets import DWTForward, DWTInverse


def wavelet_enhance(image_tensor: torch.Tensor,
                    brightness_uniformity: bool = True,
                    edge_enhancement: bool = True,
                    brightness_factor: float = 1.2,
                    edge_factor: float = 1.5,
                    preserve_contrast: bool = True) -> torch.Tensor:
    """
    ウェーブレット変換を用いた画像強調と明度均一化
    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (NCHW形式)
        brightness_uniformity (bool): 明度均一化を有効にするか
        edge_enhancement (bool): エッジ強調を有効にするか
        brightness_factor (float): 明度均一化の強度 (1.0-2.0推奨)
        edge_factor (float): エッジ強調の強度
        preserve_contrast (bool): コントラストを保持するか
    Returns:
        torch.Tensor: 処理済み画像テンソル
    """
    # 2D離散ウェーブレット変換 (DWT) を設定
    # 'db1'はHaarウェーブレット
    device = image_tensor.device
    xfm = DWTForward(J=1, wave='db1').to(device)
    ifm = DWTInverse(wave='db1').to(device)

    # NCHW形式にする（バッチ、チャンネル、高さ、幅）
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    # DWTの実行: 近似成分(Yl)と詳細成分(Yh)に分解
    # Yl: 全体の構造・明度、Yh: エッジやテクスチャ
    Yl, Yh = xfm(image_tensor)

    # --------------------------------
    # 明度均一化処理（新しいアプローチ）
    if brightness_uniformity:
        # 元の画像の統計情報を保存
        original_mean = image_tensor.mean()
        original_std = image_tensor.std()

        # 近似成分（Yl）の統計情報
        Yl_mean = Yl.mean()
        Yl_std = Yl.std()

        # 明度の不均一を補正する新しいアプローチ
        # 近似成分の範囲を適度に調整（極端な変更を避ける）
        Yl_min, Yl_max = Yl.min(), Yl.max()
        Yl_range = Yl_max - Yl_min

        # brightness_factorに基づいて範囲を調整
        # 1.0: 変更なし、1.2: 軽微な調整、1.5: 中程度の調整
        range_adjustment = 1.0 / brightness_factor

        # 新しい範囲を計算
        new_range = Yl_range * range_adjustment
        new_center = (Yl_min + Yl_max) / 2

        # 範囲を調整（中心を保持）
        Yl_min_new = new_center - new_range / 2
        Yl_max_new = new_center + new_range / 2

        # 近似成分を新しい範囲にマッピング
        Yl_enhanced = torch.clamp(Yl, Yl_min_new, Yl_max_new)

        # 元の範囲との比率を計算して、詳細成分も調整
        scale_factor = Yl_range / (Yl_max_new - Yl_min_new)

    else:
        Yl_enhanced = Yl
        scale_factor = 1.0

    # エッジ強調処理
    if edge_enhancement:
        # 高周波成分（Yh）のゲインを上げる
        # 明度均一化の影響も考慮
        Yh_enhanced = [h * edge_factor * scale_factor for h in Yh]
    else:
        Yh_enhanced = Yh

    # --------------------------------

    # 逆ウェーブレット変換で画像を再構築
    restored_tensor = ifm((Yl_enhanced, Yh_enhanced))

    # 明度均一化後の最終調整
    if brightness_uniformity and preserve_contrast:
        # 元の画像の統計特性を参考にして調整
        restored_mean = restored_tensor.mean()
        restored_std = restored_tensor.std()

        # 元の画像の統計特性に近づける
        target_mean = original_mean
        target_std = original_std * (1.0 / brightness_factor)  # 明度均一化の強度に応じて調整

        # 統計特性を調整
        restored_tensor = (restored_tensor - restored_mean) / \
            (restored_std + 1e-8) * target_std + target_mean

        # 極端な値を制限（0-1の範囲内に）
        restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)
    else:
        # 従来の正規化
        restored_tensor = (restored_tensor - restored_tensor.min()) / \
            (restored_tensor.max() - restored_tensor.min())

    return restored_tensor.squeeze()


def wavelet_brightness_uniformity(image_tensor: torch.Tensor,
                                  factor: float = 1.2,
                                  preserve_contrast: bool = True) -> torch.Tensor:
    """
    ウェーブレット変換を用いた明度均一化のみを行う関数
    Args:
        image_tensor (torch.Tensor): 入力画像テンソル
        factor (float): 明度均一化の強度 (1.0-2.0推奨)
        preserve_contrast (bool): コントラストを保持するか
    Returns:
        torch.Tensor: 明度均一化された画像テンソル
    """
    return wavelet_enhance(
        image_tensor,
        brightness_uniformity=True,
        edge_enhancement=False,
        brightness_factor=factor,
        preserve_contrast=preserve_contrast
    )


def wavelet_adaptive_brightness_uniformity(image_tensor: torch.Tensor,
                                           window_size: int = 32,
                                           strength: float = 0.5) -> torch.Tensor:
    """
    適応的ウェーブレット変換による明度均一化
    局所的な明度の不均一を検出して補正
    Args:
        image_tensor (torch.Tensor): 入力画像テンソル
        window_size (int): 局所処理のウィンドウサイズ
        strength (float): 補正の強度 (0.0-1.0)
    Returns:
        torch.Tensor: 明度均一化された画像テンソル
    """
    device = image_tensor.device
    xfm = DWTForward(J=2, wave='db1').to(device)  # 2レベル分解
    ifm = DWTInverse(wave='db1').to(device)

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    # 2レベル分解でより詳細な周波数成分を取得
    Yl, Yh = xfm(image_tensor)

    # 近似成分（Yl）の局所的な明度不均一を検出
    Yl_enhanced = Yl.clone()

    # 局所的な明度の平均と標準偏差を計算
    # パディングを追加
    pad_size = window_size // 2
    Yl_padded = torch.nn.functional.pad(
        Yl, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

    # 局所的な統計量を計算
    for i in range(Yl.shape[2]):
        for j in range(Yl.shape[3]):
            # 局所ウィンドウの統計量
            local_window = Yl_padded[:, :, i:i+window_size, j:j+window_size]
            local_mean = local_window.mean()
            local_std = local_window.std()

            # グローバルな統計量
            global_mean = Yl.mean()
            global_std = Yl.std()

            # 局所的な明度の偏差を計算
            local_brightness = Yl[:, :, i, j]
            brightness_deviation = (
                local_brightness - local_mean) / (local_std + 1e-8)

            # 適応的な補正を適用
            correction = brightness_deviation * \
                (global_std - local_std) * strength
            Yl_enhanced[:, :, i, j] = local_brightness - correction

    # 逆変換で画像を復元
    restored_tensor = ifm((Yl_enhanced, Yh))

    # 値を0-1に制限
    restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)

    return restored_tensor.squeeze()
