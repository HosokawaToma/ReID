import torch
from pytorch_wavelets import DWTForward, DWTInverse


def wavelet_enhance(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    ウェーブレット変換を用いた画像強調
    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (NCHW形式)
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
    # Yl: 全体の構造、Yh: エッジやテクスチャ
    Yl, Yh = xfm(image_tensor)

    # --------------------------------
    # 処理部分: ここで詳細成分を強調する
    # 高周波成分（Yh）のゲインを上げる
    Yh_enhanced = [h * 2.0 for h in Yh]

    # --------------------------------

    # 逆ウェーブレット変換で画像を再構築
    restored_tensor = ifm((Yl, Yh_enhanced))

    # 値を0-1に正規化して返す
    restored_tensor = (restored_tensor - restored_tensor.min()) / \
        (restored_tensor.max() - restored_tensor.min())
    return restored_tensor.squeeze()
