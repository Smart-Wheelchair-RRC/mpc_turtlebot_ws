# rrc_turtlebot_ws

In Every Terminal (add this to bashrc, make sure path to repository is correct)
```bash
source $HOME/rrc_turtlebot_ws/devel/setup.bash
export TURTLEBOT3_MODEL=burger
export HOUSEMAP=$HOME/rrc_turtlebot_ws/src/turtlebot3/turtlebot3_navigation/maps/turtlebot3_house/map.yaml
```

Terminal 1
```bash
roscore
```

Terminal 2
```bash
roslaunch turtlebot3_gazebo turtlebot3_house.launch
```

Terminal 3
```bash
rosrun turtlebot3_gazebo move_boxes.py
```

Terminal 4
```bash
roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOUSEMAP
```
