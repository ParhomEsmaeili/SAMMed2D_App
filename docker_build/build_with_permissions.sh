#!/bin/bash
docker_tag=testing:sammed2dv2
#docker push ${docker_tag}
docker build . -f Dockerfile_With_Permissions \
 -t ${docker_tag} \
 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --network=host
#docker push ${docker_tag}
