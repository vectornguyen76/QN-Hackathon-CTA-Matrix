## Docker
### Build docker
```
docker build -t vectornguyen76/cta-matrix .
```
### Run container
```
docker run --name cta-matrix -p 6000:8000 vectornguyen76/cta-matrix
```
### Rerun container
```
docker start cta-matrix
```
### Stop container
```
docker stop cta-matrix
```
### Push to hub
```
docker push vectornguyen76/cta-matrix:latest
```
### Save file rar
```
docker save -o challenge2_CTAMatrix.tar vectornguyen76/cta-matrix:latest
```
### Remove image
```
docker image rm --force vectornguyen76/cta-matrix:latest
```

## Postman
<p align="center">
  <img src="../cap_results/test_postman.jpg" alt="animated" />
</p>
