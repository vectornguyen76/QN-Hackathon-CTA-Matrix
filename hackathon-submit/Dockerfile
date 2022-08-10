# STEP 1: Pull python image
FROM python:3.9.13

# STEP 2,3: CREATE WORK DIR AND COPY FILE TO WORK DIR
WORKDIR /cta_matrix
COPY requirements.txt /cta_matrix

# Install java
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

# STEP 4,5,6: INSTALL NECESSARY PACKAGE
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip install gdown

# STEP 7: Download file weight
RUN gdown "https://drive.google.com/uc?export=download&id=1-4TlJG_3auyI1aAKL3BTLYTaXLGOlCW2"

# STEP 8: RUN COMMAND
COPY . /cta_matrix
CMD ["python", "./app.py"]