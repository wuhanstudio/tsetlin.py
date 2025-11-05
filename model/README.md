## NILMTK Model

| Model                      | Sample Rate | Sliding Window    | Comment     | Paper                                                        |
| -------------------------- | ----------- | ----------------- | ----------- | ------------------------------------------------------------ |
| FHMM                       | Every 1min  | 60 samples               | State Based | [Link](https://ieeexplore.ieee.org/document/10961750) |
| Combinatorial Optimization | Every 1h    | 40 samples               | State Based | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9714495) / [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4655131) / [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8370743&utm_source=sciencedirect_contenthosting&getft_integrator=sciencedirect_contenthosting&tag=1)|
|                            |             |                   |             |                                                              |
| KNN / SVM                  | 100kHz      | 100ms (10k samples)             | ML Based    | [Link](https://ieeexplore.ieee.org/document/9426443)         |
| Auto Encoder (DNN)         | Every 3s    | Microwave: 128 samples    | ML Based    | [Link](https://arxiv.org/abs/1912.00759)                     |
|                            |             | Fridge: 496 samples       |             |                                                              |
|                            |             | Dish Washer: 2304 samples |             |                                                              |

Summary:

- Combinatorial Optimization requires a solver, which is not available for MCU.
- Machine-Learning based approaches required a predefined large sliding window.
- The accuracy of machine learning models are very sensitive to the size of sliding windows.



Plan:

- Most datasets supported by NILMTK only provide **low-frequency** data.  

<br/>

- ML approaches require a large window size to capture the pattern of an appliance, while state-based approaches don't.  
- However, state-based approaches require state-transition models for each appliance.  

<br/>

- Combine state + learning?  



## Online Test-Time Adaptation (TTA)

- Review (2024): https://arxiv.org/pdf/2303.15361
- https://github.com/tim-learn/awesome-test-time-adaptation/blob/main/TTA-OTTA.md

| Method                                                       | Year | Website                                        | Paper                                               |
| ------------------------------------------------------------ | ---- | ---------------------------------------------- | --------------------------------------------------- |
| Online Test-Time Training (TTT)                              | 2019 | https://yueatsprograms.github.io/ttt/home.html | [link]()                                            |
| Test-Time Classifier Adjustment (T3A)                        | 2021 | https://github.com/matsuolab/T3A               | [link](https://openreview.net/forum?id=e_yvNqkJKAW) |
| Tent: Fully Test-time Adaptation by Entropy Minimization (Tent) | 2021 | https://github.com/DequanWang/tent             | [link](https://arxiv.org/abs/2006.10726)            |
| CoTTA: Continual Test-Time Adaptation (CoTTA)                | 2022 | https://github.com/qinenergy/cotta             | [link](https://arxiv.org/abs/2203.13591)            |
| Efficient Test-Time Model Adaptation without Forgetting (EATA) | 2022 | https://github.com/mr-eggplant/EATA            | [link](https://arxiv.org/abs/2204.02610)            |
| CD-TTA: Compound Domain Test-time Adaptation for Semantic Segmentation  (CD-TTA) | 2022 |                                                | [link](https://arxiv.org/pdf/2212.08356v1)          |
| Parameter-free Online Test-time Adaptation (LAME)            | 2022 | https://github.com/fiveai/LAME                 | [link](https://arxiv.org/abs/2201.05718)            |
| Robust Continual Test-time Adaptation Against Temporal Correlation (NOTE) | 2022 | https://github.com/TaesikGong/NOTE             | [link](https://arxiv.org/abs/2208.05117)            |
| DLTTA: Dynamic Learning Rate for Test-time Adaptation on Cross-domain Medical Images | 2022 | https://github.com/med-air/DLTTA               | [link]()                                            |
| DELTA: degradation-free fully test-time adaptation (DELTA)   | 2022 | https://github.com/med-air/DLTTA               | [link](https://arxiv.org/pdf/2301.13018)            |
| A probabilistic framework for lifelong test-time adaptation (PETAL) | 2023 | https://github.com/dhanajitb/petal             | [link](https://arxiv.org/abs/2212.09713)            |

