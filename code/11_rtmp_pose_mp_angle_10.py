#Добавлен вывод выходного потока
import ffmpeg
import cv2
import mediapipe as mp
import os
import numpy as np
import pickle
import csv
import requests
import signal
import sys
import logging

# Остановка процесса по сигналу SIGTERM
def handle_sigterm(signum, frame):
    logging.info("Получен сигнал SIGTERM. Завершение работы...")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

# Получаем текущий публичный IP-адрес
def get_public_ip():
    try:
        # Запрашиваем публичный IP-адрес через внешний сервис
        response = requests.get("https://api.ipify.org?format=json")
        response.raise_for_status()  # Проверяем успешность запроса
        return response.json().get("ip")
    except Exception as e:
        print(f"Ошибка при получении публичного IP-адреса: {e}")
        return None

# Получаем текущий публичный IP-адрес
current_ip = get_public_ip()

if current_ip:
    INPUT_RTMP_URL = f"rtmp://{current_ip}:1936/live/test1"
    OUTPUT_RTMP_URL = f"rtmp://{current_ip}:1936/live/test11"

    print("INPUT_RTMP_URL:", INPUT_RTMP_URL)
    print("OUTPUT_RTMP_URL:", OUTPUT_RTMP_URL)
else:
    print("Не удалось определить публичный IP-адрес.")

#name_atlet = input ('Введите имя атлета: ')
#name_atlet = 'TEST_APP'

import sys

if len(sys.argv) > 1:
    name_atlet = sys.argv[1]  # Если передано как аргумент командной строки
else:
    print("Имя атлета не указано.")
    sys.exit(1)

print(f"Имя атлета: {name_atlet}")

print(f"Имя атлета: {name_atlet}")


# Настройки RTMP
#INPUT_RTMP_URL = "rtmp://158.160.88.125:1936/live/test1"  # Входной поток
#OUTPUT_RTMP_URL = "rtmp://158.160.88.125:1936/live/test11"  # Выходной поток
OUTPUT_FILE = f"./frames/{name_atlet}_output.mp4"  # Файл для сохранения видео
SAVE_FRAMES_DIR = f"./frames/{name_atlet}_frames_predict"  # Директория для сохранения кадров с людьми

# Создаем директорию для сохранения кадров
os.makedirs(SAVE_FRAMES_DIR, exist_ok=True)

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils  # Для отрисовки скелета

# Открываем входной поток через FFmpeg
input_stream = (
    ffmpeg
    .input(INPUT_RTMP_URL)
    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
    .run_async(pipe_stdout=True)
)

# Получаем размеры кадра из входного потока
probe = ffmpeg.probe(INPUT_RTMP_URL)
video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
width = int(video_info['width'])
height = int(video_info['height'])
fps = eval(video_info['r_frame_rate'])

# Создаем выходной поток для записи в файл
output_file_process = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
    .output(OUTPUT_FILE, vcodec='libx264', acodec='aac', preset='medium', f='mp4')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

# Создаем выходной поток для передачи в RTMP
output_rtmp_process = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
    .output(OUTPUT_RTMP_URL, vcodec='libx264', format='flv', preset='ultrafast')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

# Загрузка обученной модели, scaler и label_encoder
model_path = "random_forest_model_9_no_rei.pkl"
scaler_path = "scaler_9_no_rei.pkl"
label_encoder_path = "label_encoder_9_no_rei.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)  

    # Путь для сохранения CSV-файла
csv_output_path = f'./frames/{name_atlet}_frame_predictions.csv'

# Порог вероятности для классификации
PROBABILITY_THRESHOLD = 0.25

# Открытие CSV-файла для записи
csv_file = open(csv_output_path, mode='w', newline='')  # Открываем файл вне блока `with`
csv_writer = csv.writer(csv_file)
# Запись заголовка CSV
csv_writer.writerow(['Frame Number', 'Predicted Class', 'Probability'])

# Функция для вычисления угла между тремя точками
def calculate_angle(a, b, c):
    """
    Вычисляет угол между тремя точками a, b, c.
    Точка b является центром угла.
    """
    # Векторы от точки b к точкам a и c
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    # Косинус угла через скалярное произведение
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Ограничение значения косинуса
    return np.degrees(angle)  # Преобразуем радианы в градусы

# Список индексов ключевых точек MediaPipe, которые нас интересуют
required_indices = list(range(11, 17)) + list(range(23, 29))

frame_count = 0
try:
    while True:
        # Читаем кадр из входного потока
        #print(f"Начинаем обработку входного RTMP-потока: {INPUT_RTMP_URL}")
        raw_frame = input_stream.stdout.read(width * height * 3)
        if not raw_frame:
            break

        # Преобразуем байты в массив NumPy и создаем изменяемую копию
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
        frame = frame.copy()  # Создаем изменяемую копию

        # Обработка кадра с помощью MediaPipe Pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Если найден хотя бы один человек
        if results.pose_landmarks:
            # Отрисовка скелета на исходном кадре
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Получаем координаты ключевых точек
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Определяем bounding box вокруг человека
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for landmark in landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = max(0, min(x_min, x))  # Убедимся, что координаты в пределах кадра
                y_min = max(0, min(y_min, y))
                x_max = min(w, max(x_max, x))
                y_max = min(h, max(y_max, y))

            # Формируем словарь для быстрого доступа к точкам
            landmark_dict = {i: (landmark.x * w, landmark.y * h, landmark.z) for i, landmark in enumerate(landmarks)}

            # Вычисляем углы между ключевыми точками
            angles = [
                calculate_angle(landmark_dict.get(16, (0, 0, 0)), landmark_dict.get(14, (0, 0, 0)), landmark_dict.get(12, (0, 0, 0))),
                calculate_angle(landmark_dict.get(14, (0, 0, 0)), landmark_dict.get(12, (0, 0, 0)), landmark_dict.get(24, (0, 0, 0))),
                calculate_angle(landmark_dict.get(14, (0, 0, 0)), landmark_dict.get(12, (0, 0, 0)), landmark_dict.get(11, (0, 0, 0))),

                calculate_angle(landmark_dict.get(15, (0, 0, 0)), landmark_dict.get(13, (0, 0, 0)), landmark_dict.get(11, (0, 0, 0))),
                calculate_angle(landmark_dict.get(13, (0, 0, 0)), landmark_dict.get(11, (0, 0, 0)), landmark_dict.get(12, (0, 0, 0))),
                calculate_angle(landmark_dict.get(13, (0, 0, 0)), landmark_dict.get(11, (0, 0, 0)), landmark_dict.get(23, (0, 0, 0))),

                calculate_angle(landmark_dict.get(28, (0, 0, 0)), landmark_dict.get(26, (0, 0, 0)), landmark_dict.get(24, (0, 0, 0))),
                calculate_angle(landmark_dict.get(26, (0, 0, 0)), landmark_dict.get(24, (0, 0, 0)), landmark_dict.get(12, (0, 0, 0))),
                calculate_angle(landmark_dict.get(26, (0, 0, 0)), landmark_dict.get(24, (0, 0, 0)), landmark_dict.get(23, (0, 0, 0))),

                calculate_angle(landmark_dict.get(27, (0, 0, 0)), landmark_dict.get(25, (0, 0, 0)), landmark_dict.get(23, (0, 0, 0))),
                calculate_angle(landmark_dict.get(25, (0, 0, 0)), landmark_dict.get(23, (0, 0, 0)), landmark_dict.get(24, (0, 0, 0))),
                calculate_angle(landmark_dict.get(25, (0, 0, 0)), landmark_dict.get(23, (0, 0, 0)), landmark_dict.get(11, (0, 0, 0)))
            ]

            # Преобразуем landmarks в числовой формат
            landmarks_data = []
            for i in required_indices:
                if i in landmark_dict:
                    landmarks_data.extend(landmark_dict[i])
                else:
                    landmarks_data.extend([0, 0, 0])  # Если точка не найдена, добавляем нули

            # Добавляем углы к данным
            landmarks_data.extend(angles)

            # Нормализация данных
            landmarks_scaled = scaler.transform([landmarks_data])

            # Классификация позы
            probabilities = model.predict_proba(landmarks_scaled)[0]
            max_probability = np.max(probabilities)
            predicted_class_index = np.argmax(probabilities)

            # Проверка вероятности
            if max_probability >= PROBABILITY_THRESHOLD:
                predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
                print(f"Pose: {predicted_class}, Probability: {max_probability:.2f}")
            else:
                predicted_class = 'no detect'

            # Сохранение данных в CSV
            csv_writer.writerow([frame_count, predicted_class, max_probability])

            # Проверяем, что bounding box имеет положительные размеры
            if x_max > x_min and y_max > y_min:
                # Добавляем отступы для включения текста
                padding = 50  # Отступ в пикселях
                x_min = max(0, x_min - padding)  # Уменьшаем x_min
                y_min = max(0, y_min - padding)  # Уменьшаем y_min
                x_max = min(w, x_max + padding)  # Увеличиваем x_max
                y_max = min(h, y_max + padding)  # Увеличиваем y_max

                # Рисуем bounding box на исходном кадре
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Зеленая рамка

                # Сохраняем кадр если он не 'no detect'
                if predicted_class != 'no detect':
                    # Обрезаем изображение по скорректированному bounding box
                    cropped_image = frame[y_min:y_max, x_min:x_max]

                    # Добавляем текст на обрезанное изображение
                    text_pose = f"Pose: {predicted_class}"
                    text_prob = f"Prob: {max_probability:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_thickness = 2
                    text_color = (0, 0, 255)  # Красный цвет текста

                    # Координаты для текста на обрезанном изображении
                    text_pose_position = (10, 30)  # Название позы
                    text_prob_position = (10, 60)  # Вероятность

                    # Рисуем текст на обрезанном изображении
                    cv2.putText(cropped_image, text_pose, text_pose_position, font, font_scale, text_color, font_thickness)
                    cv2.putText(cropped_image, text_prob, text_prob_position, font, font_scale, text_color, font_thickness)

                    # Сохраняем обрезанный кадр с текстом
                    frame_filename = os.path.join(SAVE_FRAMES_DIR, f"frame_{frame_count}_{predicted_class}.jpg")
                    try:
                        cv2.imwrite(frame_filename, cropped_image)
                        print(f"Сохранен кадр с человеком: {frame_filename}")
                    except Exception as e:
                        print(f"Ошибка при сохранении кадра: {e}")

        # Передаем измененный кадр в выходной поток файла
        output_file_process.stdin.write(frame.tobytes())

        # Передаем измененный кадр в выходной RTMP-поток
        output_rtmp_process.stdin.write(frame.tobytes())

        frame_count += 1

except KeyboardInterrupt:
    print("Программа прервана пользователем.")
except Exception as e:
    print(f"Ошибка: {e}")
finally:
    # Закрываем процессы
    try:
        input_stream.terminate()  # Принудительно завершаем входной поток
        input_stream.wait()
    except Exception as e:
        print(f"Ошибка при завершении input_stream: {e}")

    try:
        output_file_process.stdin.close()
        output_file_process.terminate()  # Принудительно завершаем запись в файл
        output_file_process.wait()
    except Exception as e:
        print(f"Ошибка при завершении output_file_process: {e}")

    try:
        output_rtmp_process.stdin.close()
        output_rtmp_process.terminate()  # Принудительно завершаем RTMP-поток
        output_rtmp_process.wait()
    except Exception as e:
        print(f"Ошибка при завершении output_rtmp_process: {e}")

    # Закрываем CSV-файл
    try:
        csv_file.close()
    except Exception as e:
        print(f"Ошибка при закрытии CSV-файла: {e}")