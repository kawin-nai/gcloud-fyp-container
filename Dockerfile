FROM python:3.8

ENV PORT=80

ENV MNT_DIR /mnt

ENV BUCKET fyptest-5e73d.appspot.com

COPY requirements.txt .

COPY fyptest-5e73d-firebase-adminsdk-8zrex-99bb1b9dcc.json .

# Install system dependencies
RUN set -e; \
    apt-get update -y && apt-get install -y \
    tini \
    libgl1-mesa-glx\
    lsb-release; \
    gcsFuseRepo=gcsfuse-`lsb_release -c -s`; \
    echo "deb http://packages.cloud.google.com/apt $gcsFuseRepo main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key add -; \
    apt-get update; \
    apt-get install -y gcsfuse \
    && apt-get clean

#RUN apt-get update

#RUN apt-get install -y libgl1-mesa-glx

RUN pip install -r requirements.txt

COPY ./app ./app

RUN chmod +x /app/gcsfuse_run.sh

# Use tini to manage zombie processes and signal forwarding
# https://github.com/krallin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Pass the startup script as arguments to Tini
CMD ["/app/gcsfuse_run.sh"]

ENTRYPOINT []

CMD ["python", "./app/main.py"]