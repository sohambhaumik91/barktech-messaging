@echo off

REM Start Docker Desktop (if not already running)
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

REM Stop and remove existing container if present
docker rm -f rabbitmq >nul 2>&1

REM run the rabbitmq container image
docker run -d --hostname rabbitmq --name rabbitmq -p1883:1883 -p5672:5672 -p15672:15672 rabbitmq:4.0.9-management


REM Give RabbitMQ time to boot
timeout /t 5 >nul

REM Enable MQTT plugin
docker exec rabbitmq rabbitmq-plugins enable rabbitmq_mqtt
echo mqtt plugin enabled on the rabbitmq container

echo RabbitMQ container started.
echo Management UI: http://localhost:15672