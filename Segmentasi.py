import os
import numpy as np
from math import sqrt
from skimage import io, color, img_as_ubyte
from skimage.filters import roberts, prewitt, sobel
from scipy import ndimage

# Fungsi untuk memastikan direktori output ada
def ensure_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

# Fungsi untuk menerapkan operator Frei-Chen pada citra
def vrije_chen_edge(image):
	"""Apply a Frei-Chen like operator using the two principal masks.

	This implementation uses the commonly-cited primary Frei-Chen masks
	(with sqrt(2) coefficients) and returns the gradient magnitude.
	"""
	s2 = sqrt(2.0)
	# Kernel gradien horizontal untuk Frei-Chen
	gx = np.array([[1.0, s2, 1.0],
				   [0.0, 0.0, 0.0],
				   [-1.0, -s2, -1.0]], dtype=float)
	# Kernel gradien vertikal (transpose dari gx)
	gy = gx.T

	# Konvolusi dengan kernel horizontal
	resp_x = ndimage.convolve(image.astype(float), gx, mode='reflect')
	# Konvolusi dengan kernel vertikal
	resp_y = ndimage.convolve(image.astype(float), gy, mode='reflect')

	# Hitung magnitudo gradien
	mag = np.hypot(resp_x, resp_y)
	# Normalisasi ke rentang 0-1
	mag -= mag.min()
	if mag.max() != 0:
		mag = mag / mag.max()
	return mag

# Fungsi utama untuk memproses satu citra dengan semua metode segmentasi
def process_image(img_path, out_dir):
	# Baca citra dari path
	img = io.imread(img_path)
	# Konversi ke grayscale jika diperlukan
	if img.ndim == 3:
		gray = color.rgb2gray(img)
	else:
		# Jika sudah grayscale, pastikan dalam rentang 0-1
		gray = img.astype(float) / 255.0 if img.max() > 1 else img.astype(float)

	# Dictionary untuk menyimpan hasil dari setiap metode
	results = {}
	results['roberts'] = roberts(gray)  # Operator Roberts
	results['prewitt'] = prewitt(gray)  # Operator Prewitt
	results['sobel'] = sobel(gray)      # Operator Sobel
	results['freichen'] = vrije_chen_edge(gray)  # Operator Frei-Chen custom

	# Ambil nama dasar file tanpa ekstensi
	base = os.path.splitext(os.path.basename(img_path))[0]
	saved = []
	# Simpan setiap hasil sebagai gambar PNG
	for name, arr in results.items():
		# Konversi ke 8-bit untuk penyimpanan
		out = img_as_ubyte(arr)
		out_name = f"{base}_{name}.png"
		out_path = os.path.join(out_dir, out_name)
		io.imsave(out_path, out)
		saved.append(out_path)

	return saved

# Fungsi main untuk menjalankan proses segmentasi pada citra yang dipilih
def main():
	# Dapatkan direktori script
	here = os.path.dirname(__file__)
	images_dir = os.path.join(here, 'output_images')
	out_dir = os.path.join(images_dir, 'segmentation_results')
	ensure_dir(out_dir)

	# Daftar citra yang dipilih: landscape dan portrait dengan derau Gauss dan SP
	selected = [
		'landscape_grayscale_Gauss_std15_mean_filter.png',
		'landscape_grayscale_SP_5pct_mean_filter.png',
		'portrait_grayscale_Gauss_std15_mean_filter.png',
		'portrait_grayscale_SP_5pct_mean_filter.png',
	]

	# Buat path lengkap untuk citra yang ada
	existing = [os.path.join(images_dir, name) for name in selected]
	print('Images to process:')
	for p in existing:
		print(' -', p)

	# Proses setiap citra
	for p in existing:
		saved = process_image(p, out_dir)
		print('Saved:', saved)

if __name__ == '__main__':
	main()

