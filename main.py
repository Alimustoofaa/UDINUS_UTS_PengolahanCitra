"""
=============================================================================
UTS Pengolahan Citra Digital (PCD) - A18.1601
Runner Script - Menjalankan Soal 1 dan Soal 2
=============================================================================
"""

import sys
import os

# Tambahkan path ke sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from soal1_dehazing import main as run_soal1
from soal2_denoise import main as run_soal2


def main():
    print("\n" + "█" * 70)
    print("█  UTS PENGOLAHAN CITRA DIGITAL (PCD)                               █")
    print("█  Kode MK  : A18.1601                                              █")
    print("█  Semester  : Genap 2025/2026                                       █")
    print("█  Dosen     : M.Naufal, S.Tr.T, M.Kom                              █")
    print("█" * 70)

    print("\n\n")
    metrics_soal1 = run_soal1()

    print("\n\n")
    metrics_soal2 = run_soal2()

    print("\n\n")
    print("█" * 70)
    print("█  SEMUA PROSES SELESAI!                                             █")
    print("█  Cek folder 'output/' untuk melihat hasil                          █")
    print("█" * 70)


if __name__ == "__main__":
    main()
