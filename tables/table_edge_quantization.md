| Metric                                | Value             |
|:--------------------------------------|:------------------|
| Total Parameters (Pretraining)        | 30,838            |
| Inference Parameters (Encoder + Head) | 13,569            |
| Encoder Parameters                    | 11,456            |
| FP32 Model Disk Footprint             | 130.9 KB          |
| INT8 Quantized Model Disk Footprint   | 48.6 KB           |
| Compression Ratio                     | 2.69x             |
| FP32 Single-Flow Host Latency         | 175.6 us          |
| INT8 Single-Flow Host Latency         | 731.6 us          |
| Streaming Inference Throughput        | 255,993 flows/sec |