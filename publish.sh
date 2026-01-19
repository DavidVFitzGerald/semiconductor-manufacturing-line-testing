REGION="ap-southeast-2"
ECR_URL=439440559013.dkr.ecr.ap-southeast-2.amazonaws.com

REPO_URL=${ECR_URL}/secom-test-prediction
REMOTE_IMAGE_TAG="${REPO_URL}:prod"

LOCAL_IMAGE=secom-test-prediction

aws ecr get-login-password \
  --region ${REGION} \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

docker build --platform linux/amd64 --provenance false -t ${LOCAL_IMAGE} .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}
  --region ${REGION} \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

docker build --platform linux/amd64 --provenance false -t ${LOCAL_IMAGE} .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}
