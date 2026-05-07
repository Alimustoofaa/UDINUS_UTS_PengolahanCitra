"""
=============================================================================
UTS Pengolahan Citra Digital (PCD) - Soal 2
Perbaikan Citra dengan Noise Salt-and-Pepper
=============================================================================
Metode : Median Filter + Non-Local Means Denoising
Citra  : Pesawat terbang (Grayscale dengan Salt-and-Pepper Noise)
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
NOISY_PATH = os.path.join(BASE_DIR, "images", "soal2", "noisy.png")
REFERENCE_PATH = os.path.join(BASE_DIR, "images", "soal2", "reference.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "soal2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Fungsi-Fungsi Filtering
# ============================================================================

def apply_median_filter(image, kernel_size=5):
    """
    Menerapkan Median Filter.

    Median filter bekerja dengan mengganti setiap piksel dengan
    nilai median dari piksel-piksel tetangganya dalam jendela
    berukuran kernel_size × kernel_size.

    Sangat efektif untuk noise impulsif (salt-and-pepper) karena:
    - Tidak sensitif terhadap outlier (nilai ekstrem 0 atau 255)
    - Mempertahankan tepi lebih baik dibanding mean filter
    """
    return cv2.medianBlur(image, kernel_size)


def apply_adaptive_median_filter(image, max_kernel=7):
    """
    Adaptive Median Filter - menyesuaikan ukuran kernel
    secara dinamis berdasarkan karakteristik lokal.

    Algoritma:
    1. Mulai dengan kernel kecil (3×3)
    2. Jika median bukan impuls, lanjut ke Level B
    3. Jika median adalah impuls, perbesar kernel
    4. Ulangi sampai max_kernel tercapai
    """
    output = image.copy()
    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(image, max_kernel // 2, max_kernel // 2,
                                 max_kernel // 2, max_kernel // 2,
                                 cv2.BORDER_REFLECT)

    for i in range(h):
        for j in range(w):
            kernel_size = 3
            while kernel_size <= max_kernel:
                half = kernel_size // 2
                pi, pj = i + max_kernel // 2, j + max_kernel // 2
                window = padded[pi - half:pi + half + 1,
                               pj - half:pj + half + 1]

                z_min = float(np.min(window))
                z_max = float(np.max(window))
                z_med = float(np.median(window))
                z_xy = float(padded[pi, pj])

                # Level A: Cek apakah median bukan impuls
                if z_min < z_med < z_max:
                    # Level B: Cek apakah piksel saat ini impuls
                    if z_min < z_xy < z_max:
                        output[i, j] = z_xy  # Pertahankan nilai asli
                    else:
                        output[i, j] = z_med  # Ganti dengan median
                    break
                else:
                    kernel_size += 2

            if kernel_size > max_kernel:
                output[i, j] = z_med

    return output


def apply_nlm_denoising(image, h=10, template_size=7, search_size=21):
    """
    Non-Local Means Denoising.

    Algoritma ini membandingkan patch-patch pada seluruh citra
    (bukan hanya tetangga lokal) untuk menemukan piksel-piksel
    yang mirip, kemudian merata-ratakan berdasarkan kemiripan.

    NLM(i) = Σ w(i,j) * I(j) / Σ w(i,j)

    dimana w(i,j) = exp(-||P(i) - P(j)||² / h²)
    """
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(image, None, h,
                                         template_size, search_size)
    else:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h,
                                                template_size, search_size)


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral Filter - filter yang mempertahankan tepi.

    Menggabungkan domain filtering (kedekatan spasial) dan
    range filtering (kemiripan intensitas).
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def combined_filter_pipeline(image):
    """
    Pipeline filter kombinasi:
    1. Median Filter (3×3) untuk menghilangkan noise impulsif kasar
    2. Median Filter (5×5) untuk menghilangkan sisa noise
    3. Non-Local Means untuk denoising halus dengan preservasi detail
    """
    # Step 1: Median filter kecil untuk menghilangkan impuls
    step1 = apply_median_filter(image, kernel_size=3)

    # Step 2: Median filter lebih besar untuk menyempurnakan
    step2 = apply_median_filter(step1, kernel_size=5)

    # Step 3: NLM denoising untuk detail halus
    step3 = apply_nlm_denoising(step2, h=8, template_size=7, search_size=21)

    return step1, step2, step3


# ============================================================================
# Fungsi Evaluasi Metrik
# ============================================================================

def calculate_metrics(result_img, reference_img):
    """
    Menghitung metrik evaluasi: MSE, PSNR, SSIM, SNR
    """
    # Resize jika ukuran berbeda
    if result_img.shape != reference_img.shape:
        reference_img = cv2.resize(reference_img,
                                    (result_img.shape[1], result_img.shape[0]))

    # Pastikan grayscale
    if len(result_img.shape) == 3:
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    else:
        result_gray = result_img

    if len(reference_img.shape) == 3:
        ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference_img

    # MSE
    mse_val = mse(ref_gray, result_gray)

    # PSNR
    if mse_val > 0:
        psnr_val = psnr(ref_gray, result_gray, data_range=255)
    else:
        psnr_val = float('inf')

    # SSIM
    ssim_val = ssim(ref_gray, result_gray, data_range=255)

    # SNR
    signal_power = np.mean(ref_gray.astype(np.float64) ** 2)
    noise_power = np.mean((ref_gray.astype(np.float64) -
                           result_gray.astype(np.float64)) ** 2)
    if noise_power > 0:
        snr_val = 10 * np.log10(signal_power / noise_power)
    else:
        snr_val = float('inf')

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
    print("SOAL 2: PERBAIKAN CITRA DENGAN NOISE SALT-AND-PEPPER")
    print("Metode: Median Filter + Non-Local Means Denoising")
    print("=" * 70)

    # 1. Baca citra
    print("\n[1] Membaca citra input...")
    noisy = cv2.imread(NOISY_PATH, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(REFERENCE_PATH, cv2.IMREAD_GRAYSCALE)

    if noisy is None:
        raise FileNotFoundError(f"Citra noisy tidak ditemukan: {NOISY_PATH}")
    if reference is None:
        raise FileNotFoundError(f"Citra reference tidak ditemukan: {REFERENCE_PATH}")

    print(f"    Ukuran citra noisy    : {noisy.shape}")
    print(f"    Ukuran citra reference: {reference.shape}")

    # Resize reference agar cocok
    if noisy.shape != reference.shape:
        reference = cv2.resize(reference, (noisy.shape[1], noisy.shape[0]))
        print(f"    Reference di-resize ke: {reference.shape}")

    # 2. Deteksi tingkat noise
    print("\n[2] Analisis tingkat noise...")
    salt_count = np.sum(noisy == 255)
    pepper_count = np.sum(noisy == 0)
    total_pixels = noisy.size
    noise_ratio = (salt_count + pepper_count) / total_pixels * 100
    print(f"    Piksel Salt (255) : {salt_count} ({salt_count/total_pixels*100:.2f}%)")
    print(f"    Piksel Pepper (0) : {pepper_count} ({pepper_count/total_pixels*100:.2f}%)")
    print(f"    Estimasi noise    : ~{noise_ratio:.1f}%")

    # 3. Terapkan berbagai metode filter
    print("\n[3] Menerapkan filter...")

    # Metode 1: Median Filter 3×3
    print("    [3a] Median Filter 3×3...")
    median_3 = apply_median_filter(noisy, kernel_size=3)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "median_3x3.png"), median_3)

    # Metode 2: Median Filter 5×5
    print("    [3b] Median Filter 5×5...")
    median_5 = apply_median_filter(noisy, kernel_size=5)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "median_5x5.png"), median_5)

    # Metode 3: Non-Local Means langsung
    print("    [3c] Non-Local Means Denoising...")
    nlm_direct = apply_nlm_denoising(noisy, h=12)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "nlm_direct.png"), nlm_direct)

    # Metode 4: Combined Pipeline (Median 3→5→NLM)
    print("    [3d] Combined Pipeline (Median 3→5→NLM)...")
    step1, step2, combined = combined_filter_pipeline(noisy)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "combined_step1_med3.png"), step1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "combined_step2_med5.png"), step2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "combined_final.png"), combined)

    print("    Semua filter berhasil diterapkan.")

    # 4. Evaluasi Metrik
    print("\n[4] Menghitung metrik evaluasi...")

    results = {
        'Noisy (Original)': calculate_metrics(noisy, reference),
        'Median 3×3': calculate_metrics(median_3, reference),
        'Median 5×5': calculate_metrics(median_5, reference),
        'NLM (Direct)': calculate_metrics(nlm_direct, reference),
        'Combined (Med+NLM)': calculate_metrics(combined, reference),
    }

    for name, metrics in results.items():
        print(f"\n    --- {name} ---")
        for k, v in metrics.items():
            print(f"    {k:6s}: {v:.4f}")

    # 5. Visualisasi
    print("\n[5] Membuat visualisasi perbandingan...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Soal 2: Perbaikan Citra dengan Noise Salt-and-Pepper\n'
                 'Metode: Median Filter + Non-Local Means Denoising',
                 fontsize=16, fontweight='bold')

    images_to_show = [
        (noisy, 'Citra Noisy\n(Salt & Pepper)', None),
        (median_3, 'Median Filter 3×3', results['Median 3×3']),
        (median_5, 'Median Filter 5×5', results['Median 5×5']),
        (nlm_direct, 'NLM (Direct)', results['NLM (Direct)']),
        (combined, 'Combined Pipeline\n(Median+NLM) - BEST',
         results['Combined (Med+NLM)']),
        (reference, 'Citra Reference', None),
    ]

    for idx, (img, title, metrics) in enumerate(images_to_show):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(img, cmap='gray')
        if metrics:
            axes[row, col].set_title(
                f'{title}\nPSNR={metrics["PSNR"]:.2f}dB, '
                f'SSIM={metrics["SSIM"]:.4f}', fontsize=11)
        else:
            axes[row, col].set_title(title, fontsize=12)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "perbandingan_soal2.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("    Visualisasi tersimpan.")

    # 6. Buat tabel perbandingan metrik
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.axis('off')

    table_data = []
    for name, metrics in results.items():
        table_data.append([
            name,
            f'{metrics["MSE"]:.4f}',
            f'{metrics["PSNR"]:.4f}',
            f'{metrics["SSIM"]:.4f}',
            f'{metrics["SNR"]:.4f}'
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=['Metode', 'MSE', 'PSNR (dB)', 'SSIM', 'SNR (dB)'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(5):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight best result row
    best_row = len(table_data)  # Last row (Combined)
    for j in range(5):
        table[best_row, j].set_facecolor('#E2EFDA')

    ax2.set_title('Tabel Perbandingan Metrik Evaluasi - Soal 2',
                   fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(OUTPUT_DIR, "tabel_metrik_soal2.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    # 7. Visualisasi histogram
    print("\n[6] Membuat visualisasi histogram...")
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Perbandingan Histogram Intensitas', fontsize=14,
                   fontweight='bold')

    for idx, (img, title) in enumerate([
        (noisy, 'Noisy'),
        (combined, 'Hasil Perbaikan (Combined)'),
        (reference, 'Reference')
    ]):
        axes3[idx].hist(img.ravel(), bins=256, range=(0, 255),
                        color='steelblue', alpha=0.7)
        axes3[idx].set_title(title, fontsize=12)
        axes3[idx].set_xlabel('Intensitas Piksel')
        axes3[idx].set_ylabel('Frekuensi')
        axes3[idx].set_xlim(0, 255)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histogram_soal2.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("PROSES SOAL 2 SELESAI!")
    print(f"Semua output tersimpan di: {OUTPUT_DIR}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
