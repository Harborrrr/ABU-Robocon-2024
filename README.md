# RC2024

## Description

This repository contains the Vision Group's code for the RC2024 season, developed by the RC Division of the SPR Robotics Association at China University of Petroleum, Beijing.

During the internal development phase, we used [Gitee] for version control. The repository has now been synchronized to [Github](https://github.com/harbourrrr/ABU-Robocon-2024). For historical versions, please refer to the Gitee repository (note that these versions are outdated and no longer maintained).

The hardware used in this project includes:
- Two Orbbec Astra Pro cameras
- Two MindVision industrial cameras
- One USB to RS485 converter

**This project is based on the YOLOv5 model, utilizing OpenVINO for accelerated inference on the integrated GPU. For decision-making, Q-learning reinforcement learning is employed.**

### Project Environment

- **Python version**: 3.9.x

- **Ubuntu version**: 22.04

### Core Files

- `/newmain.py`: Main program entry point
- `/Utils/tools.py`: Library, classes, and utility functions
- `/Utils/new_vino/new_vinodetect.py`: YOLOv5 inference network accelerated with OpenVINO
- `/Utils/policy`: Silo decision-making based on Q-learning reinforcement learning

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
│   └── policy                  # Silo decision-making based on Q-learning reinforcement learning
│       ├── main.py             # Decision-making entry point
│       ├── generate_states.py  # State generation        
│       ├── test.py             # Simulation testing
│       └── train.py            # Policy model training
├── newmain.py                  # Main program entry point
├── requirements.txt            # Environment configuration
├── test.py                     # Test script
└── testmain.py                 # Test script
```

## Usage Instructions and Notes

1. Since this project involves cameras, serial ports, and other hardware, it must be connected to all the hardware to run correctly.

2. In addition to the configuration file `requirements.txt`, you also need to [install the OpenVINO toolkit](https://github.com/openvinotoolkit/openvino).

3. [Orbbec drivers and OpenNI2 library](https://vcp.developer.orbbec.com.cn/resourceCenter).

4. [MindVision drivers](https://www.mindvision.com.cn/category/software/).

5. Since the policy model includes a file larger than 100MB (`/Utils/policy/models/all_states.pickle`), Git LFS is used for storage. You need to install Git LFS to pull it correctly. [Installation instructions](https://git-lfs.github.com/).

## Open Source License

This project is licensed under the MIT License. For more details, please refer to the LICENSE file.

## Authors

- **Harbor Liu**
- **Cion Huang**

### Contributors

Additionally, we would like to thank the new team members who joined during the RC2024 season for their contributions to program debugging, model training, and other processes:

- **ZhaoYun wu**
- **Jiaheng Fan**

## Contact

For any questions or inquiries, please contact us at:

- **Harbor Liu**: harbourrr123@gmail.com

- Feel free to reach out if you have any questions about the project.




