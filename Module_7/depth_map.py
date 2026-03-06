import numpy as np
import cv2
from pathlib import Path
import time

def _load_images(filepath):
        images = list()
        cap = cv2.VideoCapture(filepath)
        i=0 #переменная отслеживает каждый третий кадр
        while cap.isOpened():
            succeed, frame = cap.read()
            if succeed:
                images.append(frame)
            else:
                cap.release()
        return np.array(images)

def check_rectification_quality(rectifiedL, rectifiedR, num_points=100):
    """
    Проверка качества ректификации путем поиска соответствующих точек

    Параметры:
    rectifiedL, rectifiedR: ректифицированные изображения
    num_points: количество точек для проверки
    """
    # Преобразуем в оттенки серого если нужно
    if len(rectifiedL.shape) == 3:
        grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)
    else:
        grayL = rectifiedL
        grayR = rectifiedR

    # ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ для goodFeaturesToTrack
    max_corners = 500  # Максимальное количество углов
    quality_level = 0.0001  # Очень низкий порог качества (было 0.01)
    min_distance = 5  # Минимальное расстояние между углами (было 10)
    block_size = 3  # Размер блока для вычисления производных
    use_harris = True  # Использовать детектор Harris
    k = 0.04  # Параметр Harris

    # Находим хорошие точки для отслеживания на левом изображении
    featuresL = cv2.goodFeaturesToTrack(grayL,
        max_corners,
        quality_level,
        min_distance,
        blockSize=block_size,
        useHarrisDetector=use_harris,
        k=k)

    if featuresL is None:
        print("Не удалось найти достаточно точек для проверки")
        return

    # Находим соответствующие точки на правом изображении с помощью оптического потока
    featuresR, status, _ = cv2.calcOpticalFlowPyrLK(grayL, grayR, featuresL, None)

    # Создаем визуализацию
    visl = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
    visr = cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR)
    vis = np.hstack((visl, visr))

    good_points = 0
    total_y_diff = 0

    for i, (ptL, ptR) in enumerate(zip(featuresL, featuresR)):
        if status[i] == 1:
            xL, yL = ptL.ravel()
            xR, yR = ptR.ravel()

            # Рисуем точки
            cv2.circle(vis, (int(xL), int(yL)), 5, (0, 255, 0), -1)
            cv2.circle(vis, (int(xR) + grayL.shape[1], int(yR)), 5, (0, 255, 0), -1)

            # Рисуем линию, соединяющую соответствующие точки
            cv2.line(vis, (int(xL), int(yL)),
                    (int(xR) + grayL.shape[1], int(yR)), (0, 255, 0), 1)

            # Проверяем разницу по y (должна быть близка к 0 при хорошей ректификации)
            y_diff = abs(yL - yR)
            total_y_diff += y_diff
            good_points += 1

    h, w = vis.shape[:2]
    vis_new = cv2.resize(vis, (w//2, h//2))

    if good_points > 0:
        avg_y_diff = total_y_diff / good_points
        print(f"Проверка качества ректификации:")
        print(f"  Найдено соответствующих точек: {good_points}")
        print(f"  Средняя разница по y: {avg_y_diff:.2f} пикселей")

        if avg_y_diff < 1.0:
            print("  ✓ Отличное качество ректификации!")
        elif avg_y_diff < 3.0:
            print("  ✓ Хорошее качество ректификации")
        else:
            print("  ⚠ Среднее качество ректификации")

    # Отображаем результат
    cv2.imshow("Rectification Quality Check", vis_new)
    cv2.waitKey(1)

    return avg_y_diff

def load_calibration_params(calib_file):
    """
    Загрузка параметров калибровки стереопары из YAML файла

    Параметры:
    calib_file: путь к YAML файлу с параметрами калибровки

    Возвращает:
    Словарь с параметрами калибровки для левой и правой камер
    """
    # Загружаем YAML файл с помощью OpenCV
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)

    if not fs.isOpened():
        raise IOError(f"Не удалось открыть файл калибровки: {calib_file}")

    # Извлекаем параметры
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    r = fs.getNode("r").mat()  # Вектор Родригеса (углы поворота)
    t = fs.getNode("t").mat()  # Вектор переноса
    sz_node = fs.getNode("sz")  # Размер изображения
    image_size = [int(sz_node.at(0).real()), int(sz_node.at(1).real())]

    # Преобразуем вектор Родригеса в матрицу поворота (3x3)
    # Для стереопары у нас есть только одно значение r, предполагаем что это поворот правой камеры относительно левой
    R, _ = cv2.Rodrigues(r)

    # Важно: Для правильной работы stereoRectify, левая камера должна иметь 
    # единичную матрицу поворота и нулевой вектор переноса,
    # а правая камера - матрицу поворота R и вектор переноса T

    calib_params = {
        'camera_matrix': K,
        'dist_coeffs': D,
        'R': R,  # Матрица поворота правой камеры относительно левой
        'T': t,  # Вектор переноса правой камеры относительно левой
        'image_size': image_size,
        'rotation_vector': r,  # Исходный вектор Родригеса
        'translation_vector': t
    }
    fs.release()
    return calib_params

def preprocess_image(image):
    # Произведите необходимую предварительную обработку изображения
    # (например, сглаживание, устранение шума и т. д.)
    # Сглаживание изображения с помощью фильтра Гаусса
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # В этом примере просто преобразуем изображение в оттенки серого
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Улучшение контраста изображения
    image = cv2.equalizeHist(image)
    return image

def load_relative_param(calib_file:str) ->dict:
    # Загружаем YAML файл с помощью OpenCV
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)

    if not fs.isOpened():
        raise IOError(f"Не удалось открыть файл калибровки: {calib_file}")

    # Извлекаем параметры
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    fs.release()
    calib = dict()
    calib["R_rel"] = R
    calib["T_rel"] = T
    return calib


def compute_relative_pose(R_abs_left, T_abs_left, R_abs_right, T_abs_right) ->dict:
    """
    Вычисление относительного положения правой камеры относительно левой
    из абсолютных параметров положения камер

    Параметры:
    R_abs_left: матрица поворота левой камеры в абсолютной СК (3x3)
    T_abs_left: вектор переноса левой камеры в абсолютной СК (3x1)
    R_abs_right: матрица поворота правой камеры в абсолютной СК (3x3)
    T_abs_right: вектор переноса правой камеры в абсолютной СК (3x1)

    Возвращает:
    R_rel: матрица поворота правой камеры относительно левой (3x3)
    T_rel: вектор переноса правой камеры относительно левой (3x1)
    """

    # Преобразование точки из системы координат правой камеры в абсолютную СК:
    # P_abs = R_abs_right * P_right + T_abs_right

    # Преобразование точки из абсолютной СК в систему координат левой камеры:
    # P_left = R_abs_left^T * (P_abs - T_abs_left)

    # Подставляем первое уравнение во второе:
    # P_left = R_abs_left^T * (R_abs_right * P_right + T_abs_right - T_abs_left)
    # P_left = (R_abs_left^T * R_abs_right) * P_right + R_abs_left^T * (T_abs_right - T_abs_left)

    # Таким образом:
    R_rel = R_abs_left.T @ R_abs_right
    T_rel = R_abs_left.T @ (T_abs_right - T_abs_left)

    calib = dict()
    calib["R_rel"] = R_rel
    calib["T_rel"] = T_rel
    return calib

def visualize_epipolar_lines(imgL, imgR, num_lines=20):
    """
    Функция для визуализации эпиполярных линий на паре изображений
    Параметры:
    imgL, imgR - исходные изображения (левое и правое)
    num_lines - количество отображаемых линий
    """
    # Создаем копии изображений для рисования
    visL = imgL.copy()
    visR = imgR.copy()

    # Если изображения цветные, преобразуем в оттенки серого для лучшей видимости линий
    if len(visL.shape) == 3:
        grayL = cv2.cvtColor(visL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(visR, cv2.COLOR_BGR2GRAY)
        visL = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
        visR = cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR)

    height = visL.shape[0]

    # Рисуем горизонтальные линии через равные промежутки
    step = height // (num_lines + 1)

    for i in range(1, num_lines + 1):
        y = i * step
        # Выбираем случайный цвет для каждой линии
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Рисуем линии на обоих изображениях
        cv2.line(visL, (0, y), (visL.shape[1] - 1, y), color, 1)
        cv2.line(visR, (0, y), (visR.shape[1] - 1, y), color, 1)
        # Добавляем небольшие маркеры на линиях
        cv2.circle(visL, (visL.shape[1] // 4, y), 3, color, -1)
        cv2.circle(visR, (visR.shape[1] // 4, y), 3, color, -1)
    return visL, visR

def rectify_stereo_images(imgL, imgR, calib_L:dict, calib_R:dict, calib: dict):
    """
    Ректификация стереопары с использованием реальных параметров калибровки

    Параметры:
    imgL, imgR: исходные изображения
    calib_params: словарь с параметрами калибровки стереопары

    Возвращает:
    rectifiedL, rectifiedR: ректифицированные изображения
    Q: матрица репроекции для преобразования disparity в 3D
    """
    h, w = imgL.shape[:2]

    # Извлечение параметров калибровки
    camera_matrix_left = calib_L['camera_matrix']
    dist_coeffs_left = calib_L['dist_coeffs']
    camera_matrix_right = calib_R['camera_matrix']
    dist_coeffs_right = calib_R['dist_coeffs']
    R = calib['R_rel']  # Матрица поворота правой камеры относительно левой
    T = calib['T_rel']  # Вектор переноса правой камеры относительно левой

    # Размер изображения из калибровки
    calib_size = calib_L['image_size']

    print(f"Параметры калибровки для размера {calib_size}")

    # Параметр alpha определяет обрезку черных областей после ректификации
    # alpha=0 - обрезать черные области (максимальное обрезание)
    # alpha=1 - сохранить все пиксели (появляются черные области)
    alpha = 0  # Можно изменить на 1, если нужны все пиксели

    # Вычисление параметров ректификации
    print("Вычисление параметров ректификации...")
    rectL, rectR, proj_matrixL, proj_matrixR, Q, roiL, roiR = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        (w, h),  # размер изображения
        R, T,    # поворот и перенос между камерами
        alpha=alpha,
        newImageSize=(w, h)  # можно указать другой размер для выходных изображений
    )

    print(f"  ROI левой камеры: {roiL}")
    print(f"  ROI правой камеры: {roiR}")

    # Создание карт для ректификации
    print("Создание карт ректификации...")
    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        camera_matrix_left, dist_coeffs_left, rectL, proj_matrixL,
        (w, h), cv2.CV_32FC1
    )
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        camera_matrix_right, dist_coeffs_right, rectR, proj_matrixR,
        (w, h), cv2.CV_32FC1
    )

    # Применение ректификации
    print("Применение ректификации к изображениям...")
    rectifiedL = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)

    # Опционально: обрезаем до ROI для удаления черных краев
    if alpha == 0 and roiL[2] > 0 and roiL[3] > 0:
        x, y, w_roi, h_roi = roiL
        rectifiedL = rectifiedL[y:y+h_roi, x:x+w_roi]
        rectifiedR = rectifiedR[y:y+h_roi, x:x+w_roi]
        print(f"Изображения обрезаны до ROI: {w_roi}x{h_roi}")

    return rectifiedL, rectifiedR, Q, (roiL, roiR)

def visualize_stereo_pair(imgL, imgR, window_name="Stereo Pair", delay=50):
    """
    Функция для визуализации стереопары рядом
    """
    # Проверяем, что изображения имеют одинаковую высоту
    hL, wL = imgL.shape[:2]
    hR, wR = imgR.shape[:2]

    # Если изображения в оттенках серого, преобразуем в цветные для конкатенации
    if len(imgL.shape) == 2:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    if len(imgR.shape) == 2:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    # Изменяем размер правого изображения, если высоты не совпадают
    if hL != hR:
        scale = hL / hR
        new_w = int(wR * scale)
        imgR = cv2.resize(imgR, (new_w, hL))

    # Объединяем изображения горизонтально
    combined = np.hstack((imgL, imgR))

    # Добавляем разделительную линию
    cv2.line(combined, (wL, 0), (wL, hL-1), (255, 255, 255), 2)

    # Добавляем подписи
    cv2.putText(combined, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "RIGHT", (wL + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображаем
    h, w = combined.shape[:2]
    combined_new = cv2.resize(combined, (w//2, h//2))
    cv2.imshow(window_name, combined_new)
    key = cv2.waitKey(delay)
    return key


def main(dir: str):
    # Загружаем видео с левой и правой камеры
    pathL = str(Path(dir,'kem.011.001.left.avi'))
    pathR = str(Path(dir,'kem.011.001.right.avi'))
    imagesL = _load_images(pathL)
    imagesR = _load_images(pathR)
    calib_l = load_calibration_params(str(Path(dir,'calib/cam_plg_left.yml')))
    calib_r = load_calibration_params(str(Path(dir,'calib/cam_plg_righ.yml')))
    calib = load_relative_param(str(Path(dir,'calib/extrinsics.yml')))

    print(f"Загружено {len(imagesL)} кадров")

    # Параметры алгоритма стереозрения
    num_disparities = 112
    block_size = 5

    # Создание объекта для вычисления карты глубины
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    # Обработка изображений и вычисление карты глубины
    depth_maps = []
    for idx, (imgL, imgR) in enumerate(zip(imagesL, imagesR)):
        print(f"Обработка кадра {idx + 1}/{len(imagesL)}")

        # Имитация ректификации (в реальном проекте использовать calib.rectify)
        rectifiedL, rectifiedR, Q, rois = rectify_stereo_images(imgL, imgR, calib_l, calib_r, calib)
        # check_rectification_quality(rectifiedL, rectifiedR)

        # ВИЗУАЛИЗАЦИЯ 4: Эпиполярные линии на ректифицированных изображениях
        epi_rectL, epi_rectR = visualize_epipolar_lines(rectifiedL, rectifiedR, num_lines=8)
        visualize_stereo_pair(epi_rectL, epi_rectR, "Epipolar Lines on Rectified Images", 1)

        # Вычисление disparity (на ректифицированных изображениях в оттенках серого)
        gray_rectL = preprocess_image(rectifiedL)
        gray_rectR = preprocess_image(rectifiedR)

        disparity = stereo.compute(gray_rectL, gray_rectR)

        # ВИЗУАЛИЗАЦИЯ 5: Карта disparity
        disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disparity_vis = np.uint8(disparity_vis)
        disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
        cv2.imshow("Disparity Map", disparity_color)

        # Вычисление карты глубины
        depth_map = np.zeros_like(disparity, dtype=np.float32)
        depth_map[disparity > 0] = (num_disparities * block_size) / disparity[disparity > 0]
        # ВИЗУАЛИЗАЦИЯ 6: Карта глубины
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
        cv2.imshow("Depth Map", depth_color)

        depth_maps.append(depth_map)
        # Управление воспроизведением
        key = cv2.waitKey(200) & 0xFF
        if key == ord('q'):  # Выход
            break
        elif key == ord('p'):  # Пауза
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    pass

if __name__ == '__main__':
    main('/home/rinat/develop/3DCV/data/stereo/kem.011')
