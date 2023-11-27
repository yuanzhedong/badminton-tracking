xhost local:root

docker pull luxonis/depthai-library
docker run --rm \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /home/nvidia/:/home/nvidia/ \
    --device-cgroup-rule='c 189:* rmw' \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    luxonis/depthai-library:latest \
    bash -c "/home/nvidia/badminton-tracking/test.sh"    
