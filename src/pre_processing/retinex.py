import torch
import kornia
import cv2
import numpy as np

# 以前修正したSSRとMSR関数 (これらは変更なしで再利用)


def ssr_gpu(image_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    device = image_tensor.device
    epsilon = 1e-6
    log_i = torch.log1p(image_tensor.clamp(min=epsilon))
    kernel_size = int(sigma * 4) * 2 + 1
    gaussian_blur = kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size), (sigma, sigma)).to(device)
    log_l = gaussian_blur(log_i)
    log_r = log_i - log_l
    return log_r


def msr_gpu(image_tensor: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    ssr_results = [ssr_gpu(image_tensor, sigma) for sigma in sigmas]
    msr_result = torch.stack(ssr_results, dim=0).mean(dim=0)
    return msr_result

# ★★★ 新しい安定版Retinex関数 ★★★


def stable_retinex_gpu(image_tensor: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """
    HSV色空間のVチャンネルにのみMSRを適用し、色かぶりを防ぐ安定版Retinex。
    入力はRGBテンソルを想定。
    """
    epsilon = 1e-6

    # 1. RGB -> HSVに変換
    hsv_tensor = kornia.color.rgb_to_hsv(image_tensor)

    # 2. Vチャンネルを抽出 (H, S, Vがそれぞれ0, 1, 2番目のチャンネル)
    v_channel = hsv_tensor[:, 2:3, :, :]

    # 3. VチャンネルにのみMSRを適用
    msr_v_log = msr_gpu(v_channel, sigmas)

    # MSRの出力を[0, 1]に正規化
    min_val = msr_v_log.min(
        dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = msr_v_log.max(
        dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    range_val = (max_val - min_val).clamp(min=epsilon)
    enhanced_v = (msr_v_log - min_val) / range_val

    # 4. 元のH, Sチャンネルと、強調したVチャンネルを結合
    # hsv_tensor[:, 0:1, :, :] -> Hチャンネル
    # hsv_tensor[:, 1:2, :, :] -> Sチャンネル
    enhanced_hsv = torch.cat([hsv_tensor[:, 0:2, :, :], enhanced_v], dim=1)

    # 5. HSV -> RGBに戻す
    final_rgb = kornia.color.hsv_to_rgb(enhanced_hsv)

    return final_rgb.clamp(0.0, 1.0)


# --- 実行フロー ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_bgr_np = cv2.imread("your_image.jpg")
    image_rgb_np = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)

    tensor_rgb = torch.from_numpy(image_rgb_np.astype(
        np.float32)).permute(2, 0, 1) / 255.0
    tensor_rgb = tensor_rgb.unsqueeze(0).to(device)

    print(f"入力Tensor形状: {tensor_rgb.shape}")

    # 新しい安定版関数を呼び出す
    sigmas = [15, 80, 200]
    processed_tensor = stable_retinex_gpu(tensor_rgb, sigmas)

    output_np_rgb = (processed_tensor.squeeze(0).cpu().permute(
        1, 2, 0).numpy() * 255).astype(np.uint8)
    output_np_bgr = cv2.cvtColor(output_np_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("stable_retinex_result.jpg", output_np_bgr)

    print("安定版Retinex処理が完了しました。")
