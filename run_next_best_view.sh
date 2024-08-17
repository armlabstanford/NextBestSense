# TBD

docker run --privileged -it --rm \
        --name boshu \
        --hostname dexterous_manipulation_desktop \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix \
        -v  /home/peasant98/Desktop/Boshu:/home/user/Documents\
        --device=/dev/dri:/dev/dri \
        --device=/dev/ttyUSB0:/dev/ttyUSB0 \
        --env="DISPLAY=$DISPLAY" \
        -v /dev/video0:/dev/video0 \
        -v /dev/video1:/dev/video1 \
        -e "TERM=xterm-256color" \
        --cap-add SYS_ADMIN --device /dev/fuse \
        --gpus all -it \
        nbs:latest \
        bash