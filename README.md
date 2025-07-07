Получен полный образ виртуальной машины - ubuntu-vm.qcow2 (57,7 Гбайт) Архив (ubuntu-vm.zip - 32,7 Гбай) выложен в облако: https://cloud.mail.ru/public/xVVz/7Rhn3M8AM

Инструкция по запуску кода на виртуальной машине

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

