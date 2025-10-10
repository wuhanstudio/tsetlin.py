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
