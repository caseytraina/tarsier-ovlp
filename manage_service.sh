#!/bin/bash

case "$1" in
    start)
        echo "Starting Tarsier Chat service..."
        docker start tarsier-chat || docker run -d \
            --name tarsier-chat \
            --gpus all \
            -p 8000:8000 \
            --restart unless-stopped \
            tarsier-chat
        ;;
    stop)
        echo "Stopping Tarsier Chat service..."
        docker stop tarsier-chat
        ;;
    restart)
        echo "Restarting Tarsier Chat service..."
        docker restart tarsier-chat
        ;;
    logs)
        echo "Showing logs..."
        docker logs -f tarsier-chat
        ;;
    status)
        echo "Service status:"
        docker ps | grep tarsier-chat
        ;;
    rebuild)
        echo "Rebuilding and restarting service..."
        docker stop tarsier-chat
        docker rm tarsier-chat
        docker build -t tarsier-chat .
        docker run -d \
            --name tarsier-chat \
            --gpus all \
            -p 8000:8000 \
            --restart unless-stopped \
            tarsier-chat
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|rebuild}"
        exit 1
        ;;
esac

exit 0 