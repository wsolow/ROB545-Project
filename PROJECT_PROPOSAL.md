ROB545 KDC Project Proposal
Will Solow and Jonas Kong

Visual Servoing with Kinova Gen3 Arm for block pickup 

Manipulation from RGB images is an important capability of modern manipulator robots. Depth sensors (e.g. lidar or stereo) can be noisy, and obtaining a consistently unobstructed view for depth pose estimation is not guaranteed. In contrast, humans consistently and reliably manipulate objects even with an eye closed (monocular vision). Thus, it begs the question: can a robot complete complex manipulation tasks soley from RGB images? 

For this project, we propose to implement Image Based Visual Servoing (IBVS, Weiss et. al 1985) to reach a set point above a block. From there, the robot will execute a predefined trajectory down to the block, execute a grasp, and pick the block back up. This is the main project deliverable and we will demonstrate this utility. We have two tools available for us for this project, (1) a Kinova Gen3 7 DoF arm with a RobotIQ 2F85 gripper and (2) a Mujoco simulator with the same Kinova Gen3 setup for prototyping outside of the real hardware. 

We have two tiers of deliverables for this project. (1) IBVS resolves image features to twist commands about the robot's end effector. Our primary goal is to achieve this mapping for reliable pickup in simulation and on hardware. The Kinova Gen3 Arm is set up both with ROS2 MoveIt2 Servo and exposes its own Twist command over TCP protocol. Our first objective will be to demonstrate that we can use either of these interfaces to command the robot via twist commands in the end effector frame. If this proves to be relatively straightfoward to do, we propose to (2) implement our own twist controller using the C++ bindings exposed by the Kinova API. This will enable us to send twist commands at 30-60Hz which are then resolved to joint position commands executed on the robot at 1kHz. 

There are a few different pieces where this project has value. (1) Both of us are interested in learning more about classical control, and Will is interesting in learning image-based control policies from demonstrations. By focusing on visual servoing, we will obtain a classical understanding of the field and write code that will lay a ground work for future experiments on image-based control. (2) by aiming to deploy the system on an actual robot, we will gain additional experience with robot hardware. Use of the Kinova Gen3 arm is provided by Will's advisor Dr. Sandhya Saisubramanian (IRAS lab) and he has cleared its use with her for the purpose of this class project. Given that ROS2 can be a pain sometimes, completing the second objective (C++ twist controller) will let the IRAS bypass ROS2 while still logging data in real-time (the current Kinova TCP protocol only enablers 20Hz control). This second portion will also expose both of us to some very classical control ideas using the Jacobian, low pass filtering, etc. 

In terms of relevant literature, we will be relying primarily on the original Image Based Visual Seroving (IBVS) paper published in 1985:
1. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1087296
Additionally, there is a nice survey paper we found and a more recent idea that handles motion during the pre-specified trajectory phase of the grasp:
1. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=538972
2. https://arxiv.org/pdf/2001.05650


