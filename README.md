# ROBOCON 2024

## Description

This repository contains the RC2024 code from the **CV Algorithm Group**, part of the **Robocon Division** under the **SPR Robotics Association** at **China University of Petroleum-Beijing**.

### Competition Rules

The RC2024 competition challenges teams to design fully autonomous robots capable of completing tasks such as picking up specified colored balls and strategically placing them into five frames, each holding up to 3 balls.

### Winning Condition

A team wins if **any 3 frames** satisfy the following criteria:  
1. The frame is full (3 balls).  
2. At least **2 balls** in the frame are of the team's color.  
3. The **top ball** matches the team's color.

---

## Hardware and Environment Specifications

| **Component**      | **Details**                            |
|---------------------|----------------------------------------|
| **Cameras**         | 2x Orbbec Astra Pro, 2x MindVision    |
| **Converter**       | USB-to-RS485                          |
| **Python version**  | 3.9.x                                 |
| **OS**              | Ubuntu 22.04                          |


## Core Files

- `/newmain.py`: Main program entry point.  
- `/Utils/tools.py`: Library of utility functions and classes.  
- `/Utils/new_vino/new_vinodetect.py`: YOLOv5 inference module, accelerated with OpenVINO.  
- `/Utils/policy`: Reinforcement learning module for decision-making, based on TD learning.

---

## Project Structure

```plaintext
├── Camera                      # Orbbec camera configuration files
├── docs                        # Documentation files
├── Mindvision                  # MindVision camera configuration files
├── Models                      # YOLO model files
├── OpenNI2                     # Depth camera library files
├── R1                          # Development and testing programs for R1
├── Utils                       # Libraries, classes, and utility functions
│   ├── tools.py                # Utility functions
│   ├── new_vino
│   │   └── new_vinodetect.py   # YOLOv5 inference network accelerated with OpenVINO
│   └── policy                  # Decision-making based on TD learning
│       ├── main.py             # Decision-making entry point
│       ├── generate_states.py  # State generation        
│       ├── test.py             # Simulation testing
│       └── train.py            # Policy model training
├── newmain.py                  # Main program entry point
├── requirements.txt            # Environment configuration
├── test.py                     # Test script
└── testmain.py                 # Test script
```
## Usage Instructions

1. Ensure all required hardware (cameras, serial ports, etc.) is connected for the project to function properly.  
2. Install dependencies using `requirements.txt` and additional software:  
   - [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino).  
   - [Orbbec drivers and OpenNI2 library](https://vcp.developer.orbbec.com.cn/resourceCenter).  
   - [MindVision drivers](https://www.mindvision.com.cn/category/software/).  
3. The policy model includes a file larger than 100MB (`/Utils/policy/models/all_states.pickle`), so **Git LFS** is required to pull it correctly. Install Git LFS using [these instructions](https://git-lfs.github.com/).  

---

## Open Source License

This project is licensed under the **MIT License**. For more details, refer to the `LICENSE` file.

---

## Authors

- **Harbor Liu**  
- **Cion Huang**  

---

## Contributors

We would like to thank the following team members for their significant contributions during the RC2024 season, including debugging, model training, and other critical tasks:

- **ZhaoYun Wu**  
- **Jiaheng Fan**  

---

## Contact

For questions or inquiries, feel free to reach out to us:  

- **Harbor Liu**: [harbourrr123@gmail.com]
- **Cion Huang**: [cionhuang124@gmail.com] 

We’re happy to assist with any questions about the project!
