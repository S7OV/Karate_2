# Создание ИИ-системы по оценке исполнения ката шотокан карате-до
Проект направлен на создание автоматизированной системы судейства карате, которая будет помогать оценивать соревнования по ката.

## Трехмерный анализ движений
Освоена работа с фреймворками
Pose2Sim и EasyMocap
Выполнена калибровка внутренних и внешних параметров видеокамер
для вычисления трехмерных координат
ключевых точек
анализируемых движений

## Синхронизация видео, получение 3D-модели

## Обработка RTMP-стриминга
В Yandex Cloud развернута система для обработки видеопотока в реальном времени
Реализован веб-интерфейс
для управления процессом

### Получен полный образ виртуальной машины - ubuntu-vm.qcow2 (57,7 Гбайт) Архив (ubuntu-vm.zip - 32,7 Гбай) выложен в облако: https://cloud.mail.ru/public/xVVz/7Rhn3M8AM

#### Код RTMP-стримминга
code/11_app.py/12_app.py — Flask-сервер для управления RTMP-стримингом, обработки видео и отображения результатов.

code/11_index.html/12_index.html — Веб-интерфейс с превью видео, кнопками управления и формой для ввода данных.

code/11_rtmp_pose_mp_angle_10.py — Реализация пайплайна: прием RTMP-потока, анализ углов между суставами, сохранение результатов.

#### Модель 
- model/random_forest_model_9_no_rei.pkl
- model/rscaler_9_no_rei.pkl
- model/rlabel_encoder_9_no_rei.pkl


### Инструкция по запуску кода на виртуальной машине
```
Запуск Nginx sudo /usr/local/nginx/sbin/nginx

Убедитесь, что Nginx запущен ps aux | grep nginx

Откройте статистику RTMP в браузере http://<Публичный IP сервера>:8081/stat

Перейдите в папку проекта: cd rtmp_pose

Активируйте виртуальную среду: source .env/bin/activate

Запуск кода: python app.py

Serving Flask app 'app'
Debug mode: on WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
Running on all addresses (0.0.0.0)
Running on http://127.0.0.1:5000
Running on http://10.129.0.24:5000 Press CTRL+C to quit
Restarting with stat
Debugger is active!
Debugger PIN: 144-364-051

Проверка в браузере: http://<Публичный IP сервера>:5000/
```
