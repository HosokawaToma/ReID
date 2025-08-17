import torch
import kornia


def ssr_gpu(image_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    単一スケールのRetinexをGPUで計算します。

    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (B, C, H, W), [0.0, 1.0]
        sigma (float): ガウシアンフィルタの標準偏差（ぼかしの強さ）

    Returns:
        torch.Tensor: SSR処理後のテンソル（対数領域）
    """
    device = image_tensor.device

    # ゼロ除算を避けるための微小値
    epsilon = 1e-6

    # 対数領域に変換
    # torch.log(image_tensor + epsilon) と同じだが、log(1+x)形式の方が数値的に安定
    log_i = torch.log1p(image_tensor.clamp(min=epsilon))

    # ガウシアンフィルタで照明成分を推定
    # カーネルサイズはシグマから自動計算させる (0, 0)
    gaussian_blur = kornia.filters.GaussianBlur2d(
        (0, 0), (sigma, sigma)).to(device)
    log_l = gaussian_blur(log_i)

    # 対数領域で引き算して反射成分を求める
    log_r = log_i - log_l

    return log_r

# --- 2. Multi-Scale Retinex (MSR) のGPU実装 ---


def msr_gpu(image_tensor: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """
    複数スケールのRetinexをGPUで計算します。

    Args:
        image_tensor (torch.Tensor): 入力画像テンソル (B, C, H, W), [0.0, 1.0]
        sigmas (list[float]): 使用するガウシアンフィルタのシグマ値のリスト

    Returns:
        torch.Tensor: MSR処理後のテンソル（対数領域）
    """
    # 各スケールでSSRを計算
    ssr_results = [ssr_gpu(image_tensor, sigma) for sigma in sigmas]

    # 結果をスタックして、重み付け平均（今回は単純平均）
    msr_result = torch.stack(ssr_results, dim=0).mean(dim=0)

    return msr_result

# --- 3. 結果を画像として見られるように正規化するヘルパー関数 ---


def normalize_for_display(tensor: torch.Tensor):
    """
    Retinexの出力（対数領域）を[0.0, 1.0]の範囲に正規化します。
    """
    # 各チャンネルごとに最小値と最大値で正規化
    min_val = tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # ゼロ除算を防止
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0

    return (tensor - min_val) / range_val


# --- 実行サンプル ---
if __name__ == '__main__':
    # GPUが利用可能か確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # (Batch, Channel, Height, Width) のダミーRGB画像テンソルを作成
    dummy_rgb_tensor = torch.rand(4, 3, 256, 256).to(device)
    print(f"\n入力RGBテンソルの形状: {dummy_rgb_tensor.shape}")

    # MSRに使用するシグマのリスト（小・中・大スケール）
    sigmas = [15, 80, 200]

    # MSRを適用
    msr_output = msr_gpu(dummy_rgb_tensor, sigmas)

    print(f"MSR出力テンソルの形状: {msr_output.shape}")

    # 結果を可視化のために正規化
    displayable_tensor = normalize_for_display(msr_output)

    print("MSR処理と正規化が完了しました。")
    # print(f"正規化後の最小値: {displayable_tensor.min():.2f}, 最大値: {displayable_tensor.max():.2f}")

    # この displayable_tensor を画像として保存・表示できます。
    # import torchvision
    # torchvision.utils.save_image(displayable_tensor, "msr_result.png")
