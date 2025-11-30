# Segmentasi Citra dengan Pendekatan Discontinuity

Tugas segmentasi citra menggunakan metode Roberts, Prewitt, Sobel, dan Frei-Chen pada citra grayscale dengan derau Gaussian dan salt-and-pepper.

## Persyaratan

- Python 3.7+
- Paket yang diperlukan: `pip install -r requirements.txt`

## Cara Menjalankan

1. Pastikan folder `output_images` berisi citra hasil restorasi (mean-filtered grayscale).
2. Jalankan script: `python Segmentasi.py`
3. Hasil segmentasi akan disimpan di `output_images/segmentation_results/`

## Metode Segmentasi

- **Roberts**: Operator gradien sederhana, sensitif terhadap tepi diagonal.
- **Prewitt**: Operator gradien dengan smoothing, lebih tahan terhadap noise.
- **Sobel**: Operator gradien dengan bobot, menghasilkan tepi yang lebih kuat.
- **Frei-Chen**: Operator berbasis basis ortogonal, dirancang untuk mendeteksi tepi sejati.

## Citra Input

- `portrait_grayscale_Gauss_std15_mean_filter.png` (Gaussian std=1.5)
- `portrait_grayscale_Gauss_std35_mean_filter.png` (Gaussian std=3.5)
- `portrait_grayscale_SP_5pct_mean_filter.png` (Salt-and-Pepper 5%)
- `portrait_grayscale_SP_15pct_mean_filter.png` (Salt-and-Pepper 15%)

## Output

16 citra hasil segmentasi (4 input x 4 metode).

## Analisis

- Roberts: Cepat tapi rentan noise, baik untuk tepi tajam.
- Prewitt: Lebih halus, cocok untuk citra dengan noise sedang.
- Sobel: Paling kuat untuk deteksi tepi utama, kurang sensitif noise.
- Frei-Chen: Mengurangi false positives, baik untuk analisis tepi kompleks.

## Kesimpulan

Sobel umumnya terbaik untuk aplikasi praktis karena keseimbangan antara kekuatan tepi dan ketahanan noise. Untuk noise Gaussian, preprocessing dengan Gaussian blur membantu; untuk salt-and-pepper, median filter lebih efektif sebelum segmentasi.
