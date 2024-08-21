# TBD

docker run --privileged -it \
        --name boshu \
        --hostname dexterous_manipulation_desktop \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix \
        -v  /home/arm/NBV_Boshu:/home/user/Documents\
        --device=/dev/dri:/dev/dri \
        --device=/dev/ttyUSB0:/dev/ttyUSB0 \
        --env="DISPLAY=$DISPLAY" \
        -v /dev/video0:/dev/video0 \
        -v /dev/video1:/dev/video1 \
        -e "TERM=xterm-256color" \
        --cap-add SYS_ADMIN --device /dev/fuse \
        --gpus all -it \
        peasant98/dexterous_manipulation_desktop:latest \
        bash