from flask import Flask, render_template, request, jsonify
import subprocess
import os
import requests
from flask import send_file
from werkzeug.utils import secure_filename
from os.path import join as path_join
import os

app = Flask(__name__)

# Глобальная переменная для хранения процесса
process = None

def get_public_ip():
    try:
        # Запрашиваем публичный IP-адрес через внешний сервис
        response = requests.get("https://api.ipify.org?format=json")
        response.raise_for_status()  # Проверяем успешность запроса
        return response.json().get("ip")
    except Exception as e:
        print(f"Ошибка при получении публичного IP-адреса: {e}")
        return None

@app.route('/')
def index():
    # Получаем публичный IP-адрес
    public_ip = get_public_ip()
    if not public_ip:
        public_ip = "localhost"  # Если IP не удалось получить, используем localhost
    return render_template('index.html', public_ip=public_ip)


import threading

def read_output(stream):
    for line in stream:
        print(line, end="")  # Выводим логи в консоль сервера

@app.route('/start', methods=['POST'])
def start():
    global process
    name_atlet = request.form.get('name_atlet')
    if not name_atlet:
        return jsonify({'error': 'Имя атлета не указано'}), 400
    try:
        process = subprocess.Popen(
            ['python', 'rtmp_pose_mp_angle_10.py', name_atlet],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Запускаем чтение потоков вывода в фоновом режиме
        threading.Thread(target=read_output, args=(process.stdout,), daemon=True).start()
        threading.Thread(target=read_output, args=(process.stderr,), daemon=True).start()
        return jsonify({'message': 'Процесс запущен'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop():
    global process
    if process:
        try:
            if process.poll() is None:  # Проверяем, работает ли процесс
                process.terminate()  # Отправляем сигнал завершения
                try:
                    process.wait(timeout=5)  # Ждём завершения процесса (5 секунд)
                except subprocess.TimeoutExpired:
                    print("Процесс не завершился за отведённое время. Принудительное завершение...")
                    process.kill()  # Принудительно завершаем процесс
            else:
                print("Процесс уже завершился.")
            
            # Закрываем потоки вывода
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

            process = None
            return jsonify({'message': 'Процесс остановлен'})
        except Exception as e:
            print(f"Ошибка при остановке процесса: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Процесс не запущен'}), 400

from flask import send_from_directory

@app.route('/frames/<name_atlet>')

def get_frames(name_atlet):
    frames_dir = f"./frames/{name_atlet}_frames_predict"
    if not os.path.exists(frames_dir):
        return jsonify({'error': 'Папка с кадрами не найдена'}), 404

    # Получаем список файлов в папке
    frames = [f"/frames/{name_atlet}_frames_predict/{file}" for file in os.listdir(frames_dir) if file.endswith('.jpg')]
    output_video_path = os.path.join("/frames", f"{name_atlet}_output.mp4")
    output_video = f"/frames/{name_atlet}_output.mp4" if os.path.exists(output_video_path) else None

    return jsonify({
        'frames': frames,
        'output_video': output_video
    })

@app.route('/frames/<path:filename>')
def serve_frames(filename):
    return send_from_directory('frames', filename)

@app.route('/get_video/<name>')
def get_video(name):
    # Безопасное соединение путей
    video_filename = secure_filename(f"{name}_output.mp4")
    video_path = path_join("frames", video_filename)
    
    if not os.path.exists(video_path):
        return jsonify({"error": f"Файл {video_path} не найден"}), 404
    
    try:
        return send_file(
            video_path,
            as_attachment=True,
            download_name=f"{name}_training.mp4",
            mimetype='video/mp4'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/check_video/<name>')
def check_video(name):
    # Безопасное формирование пути
    video_filename = secure_filename(f"{name}_output.mp4")
    video_path = path_join("frames", video_filename)
    
    # Дополнительная проверка безопасности пути
    if not os.path.abspath(video_path).startswith(os.path.abspath("frames")):
        return jsonify({"error": "Недопустимый путь"}), 400
    
    exists = os.path.exists(video_path)
    return jsonify({
        "exists": exists,
        "path": video_path,
        "accessible": os.access(video_path, os.R_OK) if exists else False
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)