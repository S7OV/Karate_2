#Добавлен вывод выходного потока
import ffmpeg
import cv2
import mediapipe as mp
import os
import numpy as np
import pickle
import csv

name_atlet = input ('Введите имя атлета: ')

# Настройки RTMP
INPUT_RTMP_URL = "rtmp://158.160.88.125:1936/live/test1"  # Входной поток
OUTPUT_RTMP_URL = "rtmp://158.160.88.125:1936/live/test11"  # Выходной поток
OUTPUT_FILE = f"{name_atlet}_output.mp4"  # Файл для сохранения видео
SAVE_FRAMES_DIR = f"{name_atlet}_frames_predict"  # Директория для сохранения кадров с людьми

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
model_path = "random_forest_model.pkl"
scaler_path = "scaler.pkl"
label_encoder_path = "label_encoder.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)  

    # Путь для сохранения CSV-файла
csv_output_path = f'{name_atlet}_frame_predictions.csv'

# Порог вероятности для классификации
PROBABILITY_THRESHOLD = 0.15

# Открытие CSV-файла для записи
csv_file = open(csv_output_path, mode='w', newline='')  # Открываем файл вне блока `with`
csv_writer = csv.writer(csv_file)
# Запись заголовка CSV
csv_writer.writerow(['Frame Number', 'Predicted Class', 'Probability'])

frame_count = 0
try:
    while True:
        # Читаем кадр из входного потока
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

            # Нормализация данных    
            # Преобразуем landmarks в числовой формат
            landmarks_data = []
            for landmark in landmarks:
                landmarks_data.append(landmark.x)  # x-координата
                landmarks_data.append(landmark.y)  # y-координата
                landmarks_data.append(landmark.z)  # z-координата (если используется)

            # Нормализация данных
            landmarks_scaled = scaler.transform([landmarks_data])

            # Классификация позы
            probabilities = model.predict_proba(landmarks_scaled)[0]
            max_probability = np.max(probabilities)
            predicted_class_index = np.argmax(probabilities)

            # Проверка вероятности
            if max_probability >= PROBABILITY_THRESHOLD:
                predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
                print (f"Pose: {predicted_class}, Probability: {max_probability:.2f}")
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
                '''
                cv2.putText(frame, f"Pose: {predicted_class}", (x_min, y_min + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2)  # Синий текст
                cv2.putText(frame, f"Prob: {max_probability:.2f}", (x_min, y_max - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2)  # Синий текст
                '''
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
    input_stream.wait()
    output_file_process.stdin.close()
    output_file_process.wait()
    output_rtmp_process.stdin.close()
    output_rtmp_process.wait()

    # Закрываем CSV-файл
    csv_file.close()