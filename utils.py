import os
import cv2
import numpy as np
import pandas as pd
from skimage import exposure
from scipy.stats import entropy
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.feature import canny
from skimage.morphology import disk
from skimage.util import random_noise
from skimage.filters.rank import entropy
from skimage.morphology import skeletonize
from skimage.measure import shannon_entropy
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import generic_filter, laplace
# === Yardımcı Görselleştirme Fonksiyonları ===

# Görüntüyü pencere içerisinde gösterir
def drawMatrix(inputImage, title=""):
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    # Liste değilse, listeye çevir
    if not isinstance(inputImage, (list, tuple, np.ndarray)):
        inputImage = [inputImage]
        title = [title] if isinstance(title, str) else title
    else:
        if isinstance(title, str):
            title = [title] * len(inputImage)

    n = len(inputImage)
    dpi = 100
    max_cols = 3  # En fazla 3 sütun
    cols = min(n, max_cols)
    rows = math.ceil(n / max_cols)

    sample_image = np.array(inputImage[0])
    height, width = sample_image.shape[:2]
    figsize = (width * cols / dpi, height * rows / dpi)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # axes her zaman 2D liste olsun diye düzenle
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for idx, img in enumerate(inputImage):
        row = idx // max_cols
        col = idx % max_cols
        ax = axes[row][col]

        img = np.array(img)
        cmap = 'gray' if len(img.shape) == 2 else None
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.axis('off')
        if title and idx < len(title):
            ax.set_title(title[idx])

    # Boş kalan hücreleri kapat
    for i in range(n, rows * cols):
        row = i // max_cols
        col = i % max_cols
        axes[row][col].axis('off')

    plt.tight_layout(pad=0)
    plt.show()

# Belirtilen klasördeki .pgm dosyalarını matris olarak görselleştirir
def showPgmImages(imageList, showTitles=False):
    if not isinstance(imageList, list):
        print("HATA: Girdi bir liste olmalı.")
        return
    total = len(imageList)
    if total == 0:
        print("HATA: Liste boş.")
        return
    columns = 4
    rows = (total + columns - 1) // columns
    plt.figure(figsize=(4 * columns, 4 * rows))
    for i, path in enumerate(imageList):
        if not os.path.exists(path):
            print(f"Uyarı: Dosya bulunamadı → {path}")
            continue
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"HATA: Görüntü yüklenemedi → {path}")
            continue
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, cmap='gray')
        if showTitles:
            plt.title(os.path.basename(path))
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Klasördeki tüm .pgm uzantılı dosya yollarını listeler
def listPgmFiles(folder):
    pgmPaths = []
    for file in os.listdir(folder):
        fullPath = os.path.join(folder, file)
        if os.path.isfile(fullPath) and file.lower().endswith(".pgm"):
            pgmPaths.append(fullPath)
    return pgmPaths

# Belirtilen PGM dosyasını okur ve özellikleri ile birlikte döndürür
def readPgm(filePath):
    image = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {filePath}")
    height, width = image.shape
    if image.dtype == 'uint8':
        bitDepth = 8
    elif image.dtype == 'uint16':
        bitDepth = 16
    else:
        bitDepth = -1
    return height, width, bitDepth, image

# Gri görüntünün histogramını çizer
def showHistogram(image, title="Histogram"):
    if image.dtype == np.uint8:
        toneCount = 256
        valueRange = [0, 256]
    elif image.dtype == np.uint16:
        toneCount = 65536
        valueRange = [0, 65536]
    else:
        raise ValueError(f"Desteklenmeyen bit derinliği: {image.dtype}")
    hist = cv2.calcHist([image], [0], None, [toneCount], valueRange).flatten()
    totalPixels = image.shape[0] * image.shape[1]
    plt.figure(figsize=(12, 4))
    plt.bar(range(toneCount), hist, width=1.0, color='black')
    plt.title(title)
    plt.xlabel(f"Piksel Değeri (0 - {toneCount - 1})\nToplam Piksel: {totalPixels}")
    plt.ylabel("Piksel Sayısı")
    plt.tight_layout()
    plt.show()

# Verilen görüntüyü PGM formatında diske kaydeder
def savePgm(image, folder, fileName):
    if image.dtype != np.uint8:
        raise ValueError("Görüntü 8-bit (uint8) olmalı.")
    if len(image.shape) != 2:
        raise ValueError("Görüntü 2 boyutlu (gri tonlama) olmalı.")
    os.makedirs(folder, exist_ok=True)
    fullPath = os.path.join(folder, fileName)
    with open(fullPath, 'wb') as f:
        height, width = image.shape
        f.write(bytearray(f'P5\n{width} {height}\n255\n', 'ascii'))
        f.write(image.tobytes())
    print(f"Kaydedildi: {fullPath}")

# JPG dosyalarını gri tonlamalı PGM formatına dönüştürür
def jpgToPgm(inputFolder, outputFolder):
    os.makedirs(outputFolder, exist_ok=True)
    for fileName in os.listdir(inputFolder):
        if fileName.lower().endswith(".jpg"):
            inputPath = os.path.join(inputFolder, fileName)
            image = cv2.imread(inputPath)
            if image is None:
                print(f"HATA: Görüntü yüklenemedi → {fileName}")
                continue
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            baseName = os.path.splitext(fileName)[0]
            pgmName = baseName + ".pgm"
            outputPath = os.path.join(outputFolder, pgmName)
            cv2.imwrite(outputPath, grayImage)
            print(f"{fileName} → {pgmName} olarak kaydedildi")

# === Temel Görüntü İşleme Fonksiyonları ===

# Gauss bulanıklaştırma uygular
def gaussianBlur(matrix, kernel_size=5, sigma=1.0):
    return cv2.GaussianBlur(matrix, (kernel_size, kernel_size), sigma)

# Medyan bulanıklaştırma uygular
def medianBlur(matrix, kernel_size=5):
    return cv2.medianBlur(matrix, kernel_size)

# Morfolojik işlemler (açma, kapama, erozyon, genişletme)
def morphology(matrix, operation='close', kernel_size=5, iterations=1, kernel_shape='rect', custom_kernel=None):
    if kernel_shape == 'custom':
        if custom_kernel is None:
            raise ValueError("Özel kernel belirtildi ama kernel verilmedi.")
        kernel = custom_kernel
    else:
        shape_map = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }
        if kernel_shape not in shape_map:
            raise ValueError("Geçersiz kernel şekli.")
        kernel = cv2.getStructuringElement(shape_map[kernel_shape], (kernel_size, kernel_size))
    if operation == 'dilate':
        return cv2.dilate(matrix, kernel, iterations)
    elif operation == 'erode':
        return cv2.erode(matrix, kernel, iterations)
    elif operation == 'open':
        return cv2.morphologyEx(matrix, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        return cv2.morphologyEx(matrix, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("Geçersiz işlem tipi.")

# Kenar tespiti (Sobel, Canny, Laplace)
def edgeDetection(matrix, method='sobel', T1=100, T2=200):
    if method == 'sobel':
        grad_x = cv2.Sobel(matrix, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(matrix, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(grad_x, grad_y).astype(np.uint8)
    elif method == 'canny':
        return cv2.Canny(matrix, T1, T2)
    elif method == 'laplacian':
        return cv2.Laplacian(matrix, cv2.CV_64F).astype(np.uint8)

# Basit eşikleme ile segmentasyon
def segmentation(matrix, threshold_value=128):
    _, segmented = cv2.threshold(matrix, threshold_value, 255, cv2.THRESH_BINARY)
    return segmented

# Bağlı bileşenleri çıkarır
def connectedComponents(matrix):
    return cv2.connectedComponentsWithStats(matrix, connectivity=8)

# Histogram eşitleme uygular
def histogramEqualization(matrix):
    return cv2.equalizeHist(matrix)

# Gamma düzeltmesi uygular
def gammaCorrection(matrix, gamma=1.0):
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(matrix, look_up_table)

# Fourier dönüşümünün genlik spektrumunu üretir
def fourierTransform(matrix):
    dft = cv2.dft(np.float32(matrix), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude = np.log1p(magnitude)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

# Homomorfik filtre uygular
def homomorphicFilter(matrix, radius=30, gammaL=0.5, gammaH=2.0, c=1.0):
    img_normalized = matrix.astype(np.float32) / 255.0
    log_img = np.log1p(img_normalized)
    fft_img = np.fft.fft2(log_img)
    fft_shift = np.fft.fftshift(fft_img)
    rows, cols = matrix.shape
    crow, ccol = rows // 2, cols // 2
    U, V = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    Duv = np.sqrt((U - crow) ** 2 + (V - ccol) ** 2)
    H = (gammaH - gammaL) * (1 - np.exp(-c * (Duv ** 2) / (radius ** 2))) + gammaL
    filtered_fft = fft_shift * H
    ifft_shift = np.fft.ifftshift(filtered_fft)
    img_filtered = np.fft.ifft2(ifft_shift)
    img_filtered = np.real(img_filtered)
    exp_img = np.expm1(img_filtered)
    exp_img = np.clip(exp_img, 0, 1)
    return (exp_img * 255).astype(np.uint8)

# Kontrast germe işlemi uygular
def contrastStretching(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    stretched = (matrix - min_val) / (max_val - min_val) * 255
    return stretched.astype(np.uint8)

# Görüntüyü tersler (negatif)
def invertImage(matrix, mode='gray'):
    if mode == 'binary':
        return cv2.bitwise_not(matrix)
    elif mode == 'gray':
        bit_depth = matrix.max()
        return np.clip(bit_depth - matrix, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Mod 'gray' veya 'binary' olmalı.")

# Görüntüyü yeniden boyutlandırırken oranı korur
def resizePreserveAspect(image, max_side=512, resize_threshold=2000):
    h, w = image.shape[:2]
    original_shape = (h, w)
    if max(h, w) > resize_threshold:
        scale = max_side / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale, original_shape
    else:
        return image, 1.0, original_shape

# Entropi haritası oluşturur
def myEntropy(gray_image, disk_radius=5):
    return entropy(img_as_ubyte(gray_image), disk(disk_radius))