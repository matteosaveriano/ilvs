# ILVS - Imitation Learning for Visual Servoing
This package provides code and a dataset to test imitation learning approaches on an image-based visual servoing benchmark.

It also implements two of the approaches described in [(Paolillo and Saveriano, 2022)](https://arxiv.org/pdf/xxx.pdf) to embed a visual seervoing task into stable dynamical systems.

## Demos description
- `augment_LASA_dataset.m`: a script to augment the LASA Handwritten dataset with image features.
- `demo_LASA_VS_CLFDM.m`: a demo to run [CLFDM](https://www.sciencedirect.com/science/article/pii/S0921889014000372) on the augmented LASA Handwritten dataset.
- `demo_LASA_VS_RDS.m`: a demo to run [RDS](https://ieeexplore.ieee.org/document/8594474) on the augmente LASA Handwritten dataset.

## Software Requirements
The code is developed and tested under `Ubuntu 18.04` and `Matlab2019b`.

## References
Please acknowledge the authors in any academic publication that used parts of these codes.
```
@inproceedings{paolillo2020learning,
	author = {Paolillo, A. and Saveriano, M.},
	booktitle = {IEEE International Conference on Robotics and Automation},
	title = {Learning Stable Dynamical Systems for Visual Servoing},
	year = {2022}
}
```

## Third-party material
Third-party code and dataset have been included in this repository for convenience.

- **LASA Handwritten dataset**: please acknowledge the authors in any academic publications that have made use of the LASA HandWritten dataset by citing: *S. M. Khansari-Zadeh and A. Billard, "Learning Stable Non-Linear Dynamical Systems with Gaussian Mixture Models", IEEE Transaction on Robotics, 2011*.

- **GMR**: please acknowledge the authors in any academic publications that have made use of the GMR library by citing: *S. Calinon et al., "On Learning, Representing and Generalizing a Task in a Humanoid Robot", IEEE Transactions on Systems, Man and Cybernetics, Part B., 2006*.

- **CLFDM**: please acknowledge the authors in any academic publications that have made use of the CLFDM library by citing: *S.M. Khansari-Zadeh and A. Billard, "Learning Control Lyapunov Function to Ensure Stability of Dynamical System-based Robot Reaching Motions" Robotics and Autonomous Systems, 2014*.

- **RDS**: please acknowledge the authors in any academic publications that have made use of the RDS code by citing: *M. Saveriano and D. Lee, "Incremental skill learning of stable dynamical systems" IEEE International Conference on Intelligent Robots and Systems, 2018*.

## Note
This source code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.
