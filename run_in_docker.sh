docker build --rm -t fast3stmf -f Dockerfile-jupyter .
docker run --rm -i -p 8888:8888 -e DOCKER_STACKS_JUPYTER_CMD=notebook -v "$PWD":/home/jovyan/work -w /home/jovyan/work fast3stmf
