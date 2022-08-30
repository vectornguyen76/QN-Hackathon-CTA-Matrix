FROM python:3.7.9

WORKDIR /cta_matrix

COPY ./requirements.txt /cta_matrix

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Java environment to run VnCoreNLP
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

COPY . /cta_matrix

# Run to download Automodel
RUN python test.py

CMD ["python", "app.py"]