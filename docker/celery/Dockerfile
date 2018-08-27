FROM celery

USER root
RUN apt-get update && apt-get install -y gcc unixodbc-dev
USER user

RUN pip  install --no-cache-dir --user \
    flask-socketio \
    socketio \
    gevent
