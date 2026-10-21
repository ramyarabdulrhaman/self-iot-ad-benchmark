| Model Architecture      | Params      |   Model Size (KB) |   Batch 1 Latency (us) |   Throughput (flows/sec) | Edge Viability        |
|:------------------------|:------------|------------------:|-----------------------:|-------------------------:|:----------------------|
| Isolation Forest        | N/A (Trees) |             142   |                   12.5 |                   80,000 | High                  |
| One-Class SVM           | N/A (SVs)   |             380   |                   48.7 |                   20,533 | Moderate              |
| Deep Autoencoder        | 5,029       |              20.1 |                   18.3 |                   54,644 | High                  |
| Supervised MLP          | 13,569      |              53   |                   14.8 |                   67,567 | High                  |
| Random Forest           | N/A (Trees) |            1250   |                    8.4 |                  119,047 | High                  |
| XGBoost                 | N/A (Trees) |             480   |                    6.2 |                  161,290 | High                  |
| Mean Teacher            | 13,569      |              53   |                   15.1 |                   66,225 | High                  |
| FixMatch                | 13,569      |              53   |                   15   |                   66,666 | High                  |
| VIME                    | 13,569      |              53   |                   15   |                   66,666 | High                  |
| SCARF                   | 13,569      |              53   |                   14.9 |                   67,114 | High                  |
| Self-IoT-AD (Inference) | 13,569      |              53   |                   15.2 |                   65,789 | High (Sub-ms Latency) |