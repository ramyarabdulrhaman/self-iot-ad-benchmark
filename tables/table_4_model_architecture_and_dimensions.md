| Module             | Layer / Activation                         | In / Out    | Params              | FLOPs   |
|:-------------------|:-------------------------------------------|:------------|:--------------------|:--------|
| Encoder f_theta    | Linear + BatchNorm + LeakyReLU             | 21 -> 128   | 2,816 + 256 = 3,072 | 5.4k    |
| Encoder f_theta    | Linear + BatchNorm + LeakyReLU             | 128 -> 64   | 8,256 + 128 = 8,384 | 16.4k   |
| Projector g_psi    | Linear + ReLU                              | 64 -> 64    | 4,160               | 8.2k    |
| Projector g_psi    | Linear (Normalized)                        | 64 -> 32    | 2,080               | 4.1k    |
| Decoder r_phi      | Linear + LeakyReLU                         | 64 -> 128   | 8,320               | 16.4k   |
| Decoder r_phi      | Linear                                     | 128 -> 21   | 2,709               | 5.4k    |
| Classifier c_omega | Linear + ReLU + Dropout                    | 64 -> 32    | 2,080               | 4.1k    |
| Classifier c_omega | Linear + Sigmoid                           | 32 -> 1     | 33                  | 0.1k    |
| Pretraining Total  | Encoder + Projector + Decoder + Classifier | 21 -> 32/21 | 30,838              | 60.0k   |
| Inference Total    | Encoder + Classifier Head                  | 21 -> 1     | 13,569              | 26.5k   |