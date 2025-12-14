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

# Fungsi untuk menerapkan operator Frei-Chen
def vrije_chen_edge(image):
	s2 = sqrt(2.0)

	gx = np.array([[1.0, s2, 1.0],
				   [0.0, 0.0, 0.0],
				   [-1.0, -s2, -1.0]], dtype=float)

	gy = gx.T

	resp_x = ndimage.convolve(image.astype(float), gx, mode='reflect')
	resp_y = ndimage.convolve(image.astype(float), gy, mode='reflect')

	mag = np.hypot(resp_x, resp_y)
	mag -= mag.min()
	if mag.max() != 0:
		mag = mag / mag.max()

	return mag

# Proses citra: segmentasi + perhitungan MSE
def process_image(img_path, out_dir):
	# Baca citra
	img = io.imread(img_path)

	# Konversi ke grayscale
	if img.ndim == 3:
		gray = color.rgb2gray(img)
	else:
		gray = img.astype(float) / 255.0 if img.max() > 1 else img.astype(float)

	# Hasil segmentasi (ASLI)
	results = {}
	results['roberts'] = roberts(gray)
	results['prewitt'] = prewitt(gray)
	results['sobel'] = sobel(gray)
	results['freichen'] = vrije_chen_edge(gray)

	base = os.path.splitext(os.path.basename(img_path))[0]
	saved = []

	# Nilai MSE
	mse_values = {}

	for name, arr in results.items():
		# Simpan hasil segmentasi
		out = img_as_ubyte(arr)
		out_name = f"{base}_{name}.png"
		out_path = os.path.join(out_dir, out_name)
		io.imsave(out_path, out)
		saved.append(out_path)

		# HITUNG MSE
		diff = gray - arr
		diff_squared = diff ** 2
		total_error = np.sum(diff_squared)
		M, N = gray.shape
		mse = total_error / (M * N)

		mse_values[name] = mse

	return saved, mse_values

# MAIN PROGRAM
# Tabel MSE (16 baris)
def main():
	here = os.path.dirname(__file__)
	images_dir = os.path.join(here, 'output_images')
	out_dir = os.path.join(images_dir, 'segmentation_results')
	ensure_dir(out_dir)

	# Daftar citra input
	selected = [
		'landscape_grayscale_Gauss_std15_mean_filter.png',
		'landscape_grayscale_SP_5pct_mean_filter.png',
		'portrait_grayscale_Gauss_std15_mean_filter.png',
		'portrait_grayscale_SP_5pct_mean_filter.png',
	]

	existing = [os.path.join(images_dir, name) for name in selected]

	# TABEL MSE
	print('\nTABEL MSE HASIL SEGMENTASI (DETAIL)\n')
	print('{:<45} {:<12} {:<12}'.format(
		'Nama Citra', 'Metode', 'MSE'))
	print('-' * 75)

	for p in existing:
		saved, mse = process_image(p, out_dir)

		# ISI TABEL (16 CITRA)
		for metode, nilai_mse in mse.items():
			print('{:<45} {:<12} {:<12.6f}'.format(
				os.path.basename(p),
				metode.capitalize(),
				nilai_mse
			))


if __name__ == '__main__':
	main()
