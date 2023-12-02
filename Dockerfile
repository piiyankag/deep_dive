FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY deepdive deepdive
COPY setup.py .
COPY google_credentials.json /app/credentials.json

ENV DATA_URL=https://www.dropbox.com/scl/fi/kzb60nhqkayi214xydh3g/cropped-20231130T175429Z-001.zip?rlkey=2sskmrs0zqgzh7qepr51082v1&dl=1
ENV DATA_EXTRACT_PATH=/app/raw_data
ENV DATA_PATH=/app/raw_data/cropped
ENV HOME_PATH=/usr/local/share
ENV MODEL_TARGET=gcs
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV BUCKET_NAME=deep_dive_wagon

RUN echo "using HOME_PATH: $HOME_PATH"

RUN pip install .

COPY Makefile .
RUN make reset_local_files
RUN make load_model
RUN make download_data


CMD uvicorn deepdive.api.fast:app --host 0.0.0.0 --port $PORT
