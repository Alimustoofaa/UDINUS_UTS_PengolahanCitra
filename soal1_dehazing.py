"""
=============================================================================
UTS Pengolahan Citra Digital (PCD) - Soal 1
Perbaikan Citra Berkabut (Image Dehazing)
=============================================================================
Metode : Dark Channel Prior (DCP) + Guided Filter + CLAHE
Citra  : Pemandangan jalan berkabut (haze/fog)
=============================================================================
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
    mean_squared_error as mse
)
import os

# ============================================================================
# Konfigurasi Path
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOISY_PATH = os.path.join(BASE_DIR, "images", "soal1", "noisy.png")
REFERENCE_PATH = os.path.join(BASE_DIR, "images", "soal1", "reference.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "soal1")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Fungsi-Fungsi Dark Channel Prior (DCP)
# ============================================================================

def get_dark_channel(image, patch_size=15):
    """
    Menghitung Dark Channel dari citra.
    Dark channel adalah nilai minimum pada setiap patch lokal
    di seluruh channel warna.

    J_dark(x) = min_{y ∈ Ω(x)} ( min_{c ∈ {r,g,b}} J_c(y) )

    Prinsip dasar: Pada citra outdoor tanpa kabut, setidaknya satu
    channel warna memiliki intensitas sangat rendah pada sebagian
    besar piksel (kecuali langit).
    """
    # Minimum di seluruh channel warna
    min_channel = np.min(image, axis=2)

    # Erosi morfologi untuk mencari minimum di patch lokal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel


def estimate_atmospheric_light(image, dark_channel, top_percent=0.1):
    """
    Estimasi Atmospheric Light (A) berdasarkan piksel terbright
    pada dark channel.

    Langkah:
    1. Ambil top 0.1% piksel terbright pada dark channel
    2. Cari piksel dengan intensitas tertinggi pada citra asli
       di posisi yang sama
    3. A adalah rata-rata dari piksel-piksel tersebut
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = max(int(num_pixels * top_percent / 100), 1)

    # Flatten dan cari indeks piksel terbright pada dark channel
    dark_flat = dark_channel.ravel()
    indices = np.argsort(dark_flat)[-num_brightest:]

    # Rata-rata intensitas pada posisi terbright (lebih stabil dari max)
    image_flat = image.reshape(-1, 3)
    atmospheric_light = np.mean(image_flat[indices], axis=0)

    return atmospheric_light


def estimate_transmission(image, atmospheric_light, omega=0.95, patch_size=15):
    """
    Estimasi peta transmisi menggunakan DCP.

    t(x) = 1 - ω * min_{y ∈ Ω(x)} ( min_c ( I_c(y) / A_c ) )

    Parameter omega (0 < ω ≤ 1) mempertahankan sedikit haze
    agar hasilnya terlihat natural dan memiliki kedalaman.
    """
    # Normalisasi citra terhadap atmospheric light
    normalized = image.astype(np.float64) / (atmospheric_light.astype(np.float64) + 1e-6)

    # Hitung dark channel dari citra ternormalisasi
    dark_channel_norm = get_dark_channel(normalized, patch_size)

    # Estimasi transmisi
    transmission = 1.0 - omega * dark_channel_norm

    return transmission


def guided_filter(guide, src, radius=60, epsilon=1e-3):
    """
    Guided Filter untuk me-refine peta transmisi.

    Guided filter adalah edge-preserving smoothing filter yang
    menggunakan guidance image (biasanya citra grayscale asli)
    untuk mempertahankan detail tepi saat menghaluskan peta transmisi.

    q_i = a_k * I_i + b_k, ∀i ∈ ω_k
    dimana a_k dan b_k dihitung dari statistik lokal.
    """
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)

    mean_I = cv2.boxFilter(guide, -1, (radius, radius))
    mean_p = cv2.boxFilter(src, -1, (radius, radius))
    corr_Ip = cv2.boxFilter(guide * src, -1, (radius, radius))
    corr_II = cv2.boxFilter(guide * guide, -1, (radius, radius))

    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    result = mean_a * guide + mean_b
    return result


def recover_scene(image, transmission, atmospheric_light, t0=0.1):
    """
    Memulihkan citra bebas haze menggunakan atmospheric scattering model:

    J(x) = (I(x) - A) / max(t(x), t0) + A

    Model atmosferis:
    I(x) = J(x) * t(x) + A * (1 - t(x))

    dimana:
    - I(x) = citra terobservasi (berkabut)
    - J(x) = citra scene radiance (tanpa kabut)
    - t(x) = peta transmisi medium
    - A    = atmospheric light global
    - t0   = batas bawah transmisi (menghindari amplifikasi noise)
    """
    t_clamped = np.maximum(transmission, t0)

    result = np.zeros_like(image, dtype=np.float64)
    for c in range(3):
        result[:, :, c] = (
            (image[:, :, c].astype(np.float64) - atmospheric_light[c])
            / t_clamped + atmospheric_light[c]
        )

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_clahe_color(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    pada citra berwarna melalui channel L pada ruang warna LAB.

    CLAHE membagi citra menjadi tile-tile kecil dan menerapkan
    histogram equalization pada masing-masing tile dengan pembatasan
    kontras (clip limit) untuk menghindari amplifikasi noise.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def white_balance_correction(image):
    """
    Koreksi white balance menggunakan metode Gray World Assumption.
    Asumsi: rata-rata warna seluruh citra seharusnya abu-abu netral.
    """
    result = image.astype(np.float64)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    result[:, :, 0] = result[:, :, 0] * (avg_all / (avg_b + 1e-6))
    result[:, :, 1] = result[:, :, 1] * (avg_all / (avg_g + 1e-6))
    result[:, :, 2] = result[:, :, 2] * (avg_all / (avg_r + 1e-6))

    return np.clip(result, 0, 255).astype(np.uint8)


def gamma_correction(image, gamma=1.2):
    """
    Koreksi gamma untuk menyesuaikan brightness.
    gamma > 1 = mencerahkan area gelap
    gamma < 1 = menggelapkan area terang
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                       for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# ============================================================================
# Fungsi Evaluasi Metrik
# ============================================================================

def calculate_metrics(result_img, reference_img):
    """
    Menghitung metrik evaluasi: MSE, PSNR, SSIM, SNR.

    - MSE (Mean Squared Error): rata-rata kuadrat selisih piksel.
      Semakin kecil semakin baik.
    - PSNR (Peak Signal-to-Noise Ratio): rasio sinyal puncak terhadap noise.
      Semakin besar semakin baik. Dihitung: 10*log10(MAX²/MSE)
    - SSIM (Structural Similarity Index): mengukur kesamaan struktural.
      Rentang [-1, 1], semakin mendekati 1 semakin baik.
    - SNR (Signal-to-Noise Ratio): rasio daya sinyal terhadap noise.
      Semakin besar semakin baik.
    """
    if result_img.shape != reference_img.shape:
        reference_img = cv2.resize(reference_img,
                                    (result_img.shape[1], result_img.shape[0]))

    if len(result_img.shape) == 3:
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    else:
        result_gray = result_img
        ref_gray = reference_img

    mse_val = mse(ref_gray, result_gray)
    psnr_val = psnr(ref_gray, result_gray, data_range=255)
    ssim_val = ssim(ref_gray, result_gray, data_range=255)

    signal_power = np.mean(ref_gray.astype(np.float64) ** 2)
    noise_power = np.mean((ref_gray.astype(np.float64) -
                           result_gray.astype(np.float64)) ** 2)
    snr_val = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    return {
        'MSE': mse_val,
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'SNR': snr_val
    }


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("=" * 70)
    print("SOAL 1: PERBAIKAN CITRA BERKABUT (IMAGE DEHAZING)")
    print("Metode: Dark Channel Prior (DCP) + Guided Filter + CLAHE")
    print("=" * 70)

    # 1. Baca citra
    print("\n[1] Membaca citra input...")
    noisy = cv2.imread(NOISY_PATH)
    reference = cv2.imread(REFERENCE_PATH)

    if noisy is None:
        raise FileNotFoundError(f"Citra noisy tidak ditemukan: {NOISY_PATH}")
    if reference is None:
        raise FileNotFoundError(f"Citra reference tidak ditemukan: {REFERENCE_PATH}")

    print(f"    Ukuran citra noisy    : {noisy.shape}")
    print(f"    Ukuran citra reference: {reference.shape}")

    if noisy.shape != reference.shape:
        reference = cv2.resize(reference, (noisy.shape[1], noisy.shape[0]))
        print(f"    Reference di-resize ke: {reference.shape}")

    # 2. Hitung Dark Channel
    print("\n[2] Menghitung Dark Channel...")
    noisy_float = noisy.astype(np.float64)
    dark_channel = get_dark_channel(noisy_float, patch_size=15)
    dc_vis = np.clip(dark_channel, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "dark_channel.png"), dc_vis)
    print("    Dark Channel berhasil dihitung.")

    # 3. Estimasi Atmospheric Light
    print("\n[3] Estimasi Atmospheric Light...")
    atm_light = estimate_atmospheric_light(noisy_float, dark_channel)
    print(f"    Atmospheric Light (B,G,R): {atm_light}")

    # 4. Estimasi Transmission Map
    print("\n[4] Estimasi Transmission Map...")
    transmission_raw = estimate_transmission(noisy_float, atm_light,
                                              omega=0.75, patch_size=15)

    # Refine transmission dengan guided filter
    gray_guide = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    transmission_refined = guided_filter(gray_guide, transmission_raw, radius=40, epsilon=1e-3)
    transmission_refined = np.clip(transmission_refined, 0.05, 1.0)

    trans_vis = (transmission_refined * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "transmission_map_raw.png"),
                (np.clip(transmission_raw, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "transmission_map_refined.png"), trans_vis)
    print("    Transmission Map berhasil diestimasi dan di-refine.")

    # 5. Scene Recovery
    print("\n[5] Melakukan Scene Recovery (Dehaze)...")
    dehazed = recover_scene(noisy_float, transmission_refined, atm_light, t0=0.15)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step1_dehazed_dcp.png"), dehazed)
    print("    Dehaze dengan DCP selesai.")

    # 6. White Balance Correction
    print("\n[6] Koreksi White Balance...")
    wb_corrected = white_balance_correction(dehazed)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_white_balanced.png"), wb_corrected)
    print("    White Balance selesai.")

    # 7. Enhancement dengan CLAHE
    print("\n[7] Meningkatkan kontras dengan CLAHE...")
    enhanced = apply_clahe_color(wb_corrected, clip_limit=2.0, tile_size=(8, 8))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_clahe_enhanced.png"), enhanced)
    print("    CLAHE Enhancement selesai.")

    # 8. Gamma correction
    print("\n[8] Gamma Correction...")
    final = gamma_correction(enhanced, gamma=0.9)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_result.png"), final)
    print("    Gamma correction selesai.")

    # 9. Evaluasi Metrik
    print("\n[9] Menghitung metrik evaluasi...")

    all_metrics = {}

    all_metrics['Noisy (Berkabut)'] = calculate_metrics(noisy, reference)
    print("\n    --- Metrik Citra Noisy (Sebelum Perbaikan) ---")
    for k, v in all_metrics['Noisy (Berkabut)'].items():
        print(f"    {k:6s}: {v:.4f}")

    all_metrics['DCP Only'] = calculate_metrics(dehazed, reference)
    print("\n    --- Metrik Setelah DCP ---")
    for k, v in all_metrics['DCP Only'].items():
        print(f"    {k:6s}: {v:.4f}")

    all_metrics['DCP + WB'] = calculate_metrics(wb_corrected, reference)
    print("\n    --- Metrik Setelah DCP + White Balance ---")
    for k, v in all_metrics['DCP + WB'].items():
        print(f"    {k:6s}: {v:.4f}")

    all_metrics['DCP + WB + CLAHE'] = calculate_metrics(enhanced, reference)
    print("\n    --- Metrik Setelah DCP + WB + CLAHE ---")
    for k, v in all_metrics['DCP + WB + CLAHE'].items():
        print(f"    {k:6s}: {v:.4f}")

    all_metrics['Final (+ Gamma)'] = calculate_metrics(final, reference)
    print("\n    --- Metrik Final (DCP + WB + CLAHE + Gamma) ---")
    for k, v in all_metrics['Final (+ Gamma)'].items():
        print(f"    {k:6s}: {v:.4f}")

    # 10. Visualisasi
    print("\n[10] Membuat visualisasi perbandingan...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Soal 1: Perbaikan Citra Berkabut (Image Dehazing)\n'
                 'Metode: Dark Channel Prior + Guided Filter + CLAHE',
                 fontsize=16, fontweight='bold')

    # Row 1 - Process steps
    axes[0, 0].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Citra Input (Berkabut)', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(dc_vis, cmap='gray')
    axes[0, 1].set_title('Dark Channel', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(trans_vis, cmap='gray')
    axes[0, 2].set_title('Transmission Map (Refined)', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2 - Results
    m_dcp = all_metrics['DCP Only']
    axes[1, 0].imshow(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Hasil DCP\nPSNR={m_dcp["PSNR"]:.2f}dB, '
                          f'SSIM={m_dcp["SSIM"]:.4f}', fontsize=11)
    axes[1, 0].axis('off')

    m_final = all_metrics['Final (+ Gamma)']
    axes[1, 1].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Hasil Final (DCP+WB+CLAHE+Gamma)\nPSNR={m_final["PSNR"]:.2f}dB, '
                          f'SSIM={m_final["SSIM"]:.4f}', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Citra Reference', fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "perbandingan_soal1.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    # Tabel metrik
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.axis('off')

    table_data = []
    for name, metrics in all_metrics.items():
        table_data.append([
            name,
            f'{metrics["MSE"]:.4f}',
            f'{metrics["PSNR"]:.4f}',
            f'{metrics["SSIM"]:.4f}',
            f'{metrics["SNR"]:.4f}'
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=['Tahap', 'MSE', 'PSNR (dB)', 'SSIM', 'SNR (dB)'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for j in range(5):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight best PSNR row
    psnr_values = [m['PSNR'] for m in all_metrics.values()]
    best_idx = np.argmax(psnr_values) + 1
    for j in range(5):
        table[best_idx, j].set_facecolor('#E2EFDA')

    ax2.set_title('Tabel Perbandingan Metrik Evaluasi - Soal 1',
                   fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(OUTPUT_DIR, "tabel_metrik_soal1.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    # Pipeline visualization
    fig3, axes3 = plt.subplots(1, 5, figsize=(25, 5))
    fig3.suptitle('Pipeline Perbaikan Citra Berkabut (Step-by-Step)',
                   fontsize=14, fontweight='bold')

    steps = [
        (noisy, 'Input\n(Berkabut)'),
        (dehazed, 'Step 1:\nDCP Dehaze'),
        (wb_corrected, 'Step 2:\nWhite Balance'),
        (enhanced, 'Step 3:\nCLAHE'),
        (final, 'Step 4:\nGamma Correction'),
    ]

    for idx, (img, title) in enumerate(steps):
        axes3[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes3[idx].set_title(title, fontsize=11)
        axes3[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pipeline_soal1.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    print("    Visualisasi tersimpan.")

    print("\n" + "=" * 70)
    print("PROSES SOAL 1 SELESAI!")
    print(f"Semua output tersimpan di: {OUTPUT_DIR}")
    print("=" * 70)

    return all_metrics


if __name__ == "__main__":
    main()
