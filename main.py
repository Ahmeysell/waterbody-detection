import os
import cv2
import importlib
import numpy as np
import matplotlib.pyplot as plt
import utils
importlib.reload(utils)

def FindWater(pgm_path):
    img = utils.readPgm(pgm_path)[3]
    resized_img, scale, original_shape = utils.resizePreserveAspect(img)

    # === GÖKYÜZÜ ANALİZİ PIPELINE ===
    pm1 = utils.gammaCorrection(resized_img, 5)
    p0 = utils.segmentation(pm1, 20)

    height = p0.shape[0]
    cut = int(height * 0.25)
    sky_mask = np.zeros_like(p0)
    enhance_mask = np.zeros_like(p0)
    sky_mask[:cut, :] = p0[:cut, :]

    sky_ratio_thresh = 0.75
    sky_mask_roi = sky_mask[:cut, :]
    total_sky_area = sky_mask_roi.shape[0] * sky_mask_roi.shape[1]
    num_sky_labels, sky_labels, sky_stats, _ = cv2.connectedComponentsWithStats(sky_mask_roi)
    sky_component_found = False
    for i in range(1, num_sky_labels):
        area = sky_stats[i, cv2.CC_STAT_AREA]
        if area / total_sky_area >= sky_ratio_thresh:
            sky_component_found = True
            break
    if sky_component_found:
        sky_mask[:cut, :] = 255
        enhance_mask[cut:, :] = p0[cut:, :]
    else:
        sky_mask[:cut, :] = 0
        enhance_mask[:, :] = p0[:, :]

    enhance_mask = utils.morphology(enhance_mask, operation="erode", kernel_size=5, kernel_shape="ellipse", iterations=1)
    filled_enhance_mask = np.zeros_like(enhance_mask)
    contours, _ = cv2.findContours(enhance_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(filled_enhance_mask, contours, i, 255, -1)

    area_thresh = 5000
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_enhance_mask)
    filtered_enhance_mask = np.zeros_like(enhance_mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_thresh:
            filtered_enhance_mask[labels == i] = 255

    # === KÖPÜK TESPİT PIPELINE ===
    entropy_map = utils.myEntropy(resized_img, 5)
    entropy_thresh = 5
    brightness_thresh = 150
    foam_mask = np.zeros_like(resized_img, dtype=np.uint8)
    foam_mask[(entropy_map > entropy_thresh) & (resized_img > brightness_thresh)] = 255
    enhance_foam_mask = filtered_enhance_mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foam_mask)
    for i in range(1, num_labels):
        single_foam = np.zeros_like(foam_mask, dtype=np.uint8)
        single_foam[labels == i] = 255
        intersection = cv2.bitwise_and(single_foam, filtered_enhance_mask)
        if np.any(intersection):
            enhance_foam_mask = cv2.bitwise_or(enhance_foam_mask, single_foam)

    foam_final = utils.morphology(enhance_foam_mask, operation="close", kernel_size=10, kernel_shape='ellipse', iterations=1)

    # === ANA PIPELINE ===
    p1 = utils.homomorphicFilter(resized_img, radius=60, gammaL=0.9, gammaH=2, c=1)
    p2 = utils.edgeDetection(p1, "canny", T1=100, T2=150)
    p3 = utils.invertImage(p2, "binary")
    p3[sky_mask == 255] = 0
    circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    p4 = utils.morphology(p3, operation="erode", kernel_shape='custom', custom_kernel=circle)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(p4)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + np.argmax(areas)
    main_area = areas[np.argmax(areas)]
    main_centroid = centroids[max_label]
    area_ratio_thresh = 0.25
    distance_thresh = 250.0
    p5 = np.zeros_like(p4)
    for i in range(1, num_labels):
        this_area = stats[i, cv2.CC_STAT_AREA]
        this_centroid = centroids[i]
        if i == max_label or (
            this_area / main_area >= area_ratio_thresh and
            np.linalg.norm(np.array(this_centroid) - np.array(main_centroid)) < distance_thresh
        ):
            p5[labels == i] = 255

    main_area = cv2.countNonZero(p5)
    image_area = p5.shape[0] * p5.shape[1]
    main_ratio = main_area / image_area
    num_main_labels, _, _, _ = utils.connectedComponents(p5)
    main_components = num_main_labels - 1
    support_threshold = 0.08

    if main_components > 2:
        p5 = foam_final.copy()
    elif main_components <= 2 and main_ratio < support_threshold:
        p5 = cv2.bitwise_or(p5, foam_final)

    circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    p6 = utils.morphology(p5, operation="dilate", kernel_shape='custom', custom_kernel=circle)

    contours, _ = cv2.findContours(p6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(resized_img.shape) == 2:
        p8 = cv2.cvtColor(resized_img.copy(), cv2.COLOR_GRAY2BGR)
    else:
        p8 = resized_img.copy()
    cv2.drawContours(p8, contours, -1, (255, 0, 0), thickness=2)

    contour_mask = np.zeros_like(p6)
    cv2.drawContours(contour_mask, contours, -1, color=255, thickness=2)
    contour_mask_large = cv2.resize(contour_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    if len(img.shape) == 2:
        result = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    else:
        result = img.copy()
    result[contour_mask_large == 255] = [255, 0, 0]

    # === ÇIKTILAR ===
    return {
        "Gri Görüntü": resized_img,
        "Gamma": pm1,
        "Threshold": p0,
        "Gökyüzü Maskesi": sky_mask,
        "Güçlendirme Maskesi": enhance_mask,
        "Filtreli Güçlendirme Maskesi": filtered_enhance_mask,
        "Entropi Haritası": entropy_map,
        "Köpük Maskesi": foam_mask,
        "Son Köpük Maskesi": foam_final,
        "Homomorfik Filtre": p1,
        "Canny": p2,
        "Terslenmiş Görüntü": p3,
        "Erode": p4,
        "Ana Tahmin": p5,
        "Genişletilmiş Sonuç": p6,
        "Kontur Çizimi": p8,
        "Sonuç": result
    }

'''
utils.jpgToPgm("Data/Train", "Data/Train")
utils.jpgToPgm("Data/Test", "Data/Test")
'''

image_paths = [
    "Data/Train/Kolay-1.pgm", "Data/Test/Kolay-2.pgm",
    "Data/Train/OrtaAlt-1.pgm", "Data/Test/OrtaAlt-2.pgm",
    "Data/Train/Orta-1.pgm", "Data/Test/Orta-2.pgm",
    "Data/Test/OrtaUst-1.pgm", "Data/Train/OrtaUst-2.pgm",
    "Data/Test/Zor-1.pgm", "Data/Train/Zor-2.pgm",
    "Data/Train/CokZor-1.pgm", "Data/Test/CokZor-2.pgm"
]

results = FindWater(image_paths[2])
keys = list(results.keys())

utils.drawMatrix([results[keys[0]], results[keys[-1]]], title=[keys[0], keys[-1]])

