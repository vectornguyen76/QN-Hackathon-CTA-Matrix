## HACKATHON CHALLENGE 2 DOCKER DEMO
**Folder contains the modules**
- `model.py`: This module used for loading model 
- `processing.py`: This module used for processing raw text before pass to model
- `solver.py`: This module used for classifying (**DO NOT MODIFY FUNCTION NAME (solve)**)
- `app.py`: This module used for expose API (**DO NOT MODIFY**)
- `config.py`: Containing parameters
- `setting.py`: Contains Host, Port, Info (**DO NOT MODIFY**)
- `Dockerfile`: Note: At step 7
  + Upload file_weight to google drive.
  + Set Editor permission for anyone have link.
  + Open [link](https://sites.google.com/site/gdocs2direct/) to create direct link. 
  + `MODEL_FILE_NAME` in `config.py` must same name on `google_drive`


**Guideline for duilding docker-image and upload to landing page**
- Step 1: Build docker image (Note: punc "." end of command): **docker build -t image_name .**
  + example: *docker build -t loiln/aihkt_challenge_02 .*
- Step 2: Check docker image after building
  + *docker images*
- Step 3: Run created docker image: **docker run --name container_name -p <'local-port'>:8000 image_name**
  + example: *docker run --name aihkt -p 6000:8000 loiln/aihkt_challenge_02*
- Step 4: Test result on postman: **http://127.0.0.1:<'local-port'>/review-solver/solve?review_sentence**
  + example: *http://127.0.0.1:6000/review-solver/solve?review_sentence*
  + pass url to postman -> fill raw text in parameters **review_sentence**
- Step 5: After testing postman success, wrap docker to docker-image.tar: **docker save -o image_name.tar image_name:tag** (you can get info in step 2)
  + example: *docker save -o aihkt_challenge_02.tar loiln/aihkt_challenge_02:latest*
- Step 6: Upload file docker image_name.tar to landing page: https://hackathon.quynhon.ai/challenges