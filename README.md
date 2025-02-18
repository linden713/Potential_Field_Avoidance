# Potential_Field_Avoidance
Arm Obstacle Avoidance Using Potential Filed in Isaac Lab


This script demonstrates a potential field based controller for collision-free motion 
using Isaac Lab env. Only repulsive forces are computed 
for obstacle avoidance. For each joint, the repulsive force is calculated based on its 
workspace (world) position relative to the obstacle, and then mapped to joint space using 
the joint’s Jacobian matrix.

Usage:
```python
    ./isaaclab.sh -p pf_control.py 
```
Video: 