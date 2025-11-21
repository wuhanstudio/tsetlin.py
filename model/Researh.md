# On-Device Training

|                      | ARM                                          | MCU                |
| -------------------- | -------------------------------------------- | ------------------ |
| Neural Network (NN)  | TFLite / LiteRT (IOS, Android, Raspi)        | TinyEngine / AIfES |
| Tsetlin Machine (TM) | TMU                                          |                    |
| Dataset              | Image (MNIST, Cifar-10, Cifar-100, ImageNet) | Sensor (Non-Image) |



# On-Device TTA

|                      | ARM                | MCU                        |
| -------------------- | ------------------ | -------------------------- |
| Neural Network (NN)  | BoTTA (T3A, TENT)  | TinyTTA (Not reproducible) |
| Tsetlin Machine (TM) |                    |                            |
| Dataset              | Cifar-10 Corrupted | MNIST-Corrupted            |



# NILM Application

|             | Tsetlin Machine (TM) |      |
| ----------- | -------------------- | ---- |
| Dataset     | REDD / UK-DALE       |      |
| Training    |                      |      |
| Inference   |                      |      |
| Compression |                      |      |

