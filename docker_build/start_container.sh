docker container run --gpus all --rm -ti \
--volume /home/parhomesmaeili/IS_Codebase_Forks/SAM-Med2D_Fork:/workspace/IS_Codebase_Forks/SAM-Med2D_Fork \
--volume /home/parhomesmaeili/IS-Validation-Framework:/workspace/IS-Validation-Framework \
--volume /home/parhomesmaeili/env_bashscripts/is_runscripts/sammed2d_runscript.sh:/workspace/runscripts/sammed2d_runscript.sh \
--volume /home/parhomesmaeili/local_docker_vscode-server:/home/parhomesmaeili/.vscode-server \
--cpus 16 \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--ipc host \
--name sammed2dv2 \
testing:sammed2dv2