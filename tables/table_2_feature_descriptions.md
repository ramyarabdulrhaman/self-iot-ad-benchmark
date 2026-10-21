| Feature Name         | Description                                                         |   Min |              Max |     Mean |       Std |
|:---------------------|:--------------------------------------------------------------------|------:|-----------------:|---------:|----------:|
| flow_duration        | Total duration of bidirectional flow (seconds)                      |     0 |  21728.3         |    25.34 |    327.05 |
| fwd_pkts_tot         | Cumulative packet count transmitted source-to-destination           |     0 |   4345           |     9.53 |     56.51 |
| bwd_pkts_tot         | Cumulative packet count transmitted destination-to-source           |     0 |  10112           |     8.05 |     86.78 |
| fwd_data_pkts_tot    | Number of forward packets carrying nonzero payload bytes            |     0 |   4345           |     4.47 |     49.66 |
| bwd_data_pkts_tot    | Number of backward packets carrying nonzero payload bytes           |     0 |  10105           |     5.6  |     84.97 |
| fwd_pkts_per_sec     | Forward packet rate (packets / second)                              |     0 |      1.04858e+06 |  2078.81 |  23091.1  |
| bwd_pkts_per_sec     | Backward packet rate (packets / second)                             |     0 |      1.04858e+06 |  1861.53 |  22570.4  |
| flow_pkts_per_sec    | Aggregate bidirectional packet transmission rate                    |     0 |      2.09715e+06 |  3940.35 |  45393.7  |
| down_up_ratio        | Ratio of downloaded to uploaded packet volume                       |     0 |      6.09        |     0.83 |      0.27 |
| fwd_header_size_tot  | Cumulative header length in forward direction (bytes)               |     0 |  69296           |   252.42 |   1004.69 |
| bwd_header_size_tot  | Cumulative header length in backward direction (bytes)              |     0 | 323592           |   219.68 |   2704.26 |
| fwd_pkts_payload.min | Minimum payload size among forward packets (bytes)                  |     0 |   1097           |    17.77 |     28.81 |
| fwd_pkts_payload.max | Maximum payload size among forward packets (bytes)                  |     0 |   1420           |   185.36 |    302.2  |
| fwd_pkts_payload.tot | Cumulative payload bytes sent in forward direction                  |     0 | 747340           |   863.41 |  12453.2  |
| fwd_pkts_payload.avg | Mean payload byte size of forward packets                           |     0 |   1319.37        |    46.94 |     72.72 |
| bwd_pkts_payload.min | Minimum payload size among backward packets (bytes)                 |     0 |   1357           |    24.54 |     42.56 |
| bwd_pkts_payload.max | Maximum payload size among backward packets (bytes; TSO aggregated) |     0 |   5124           |   355.66 |    510.94 |
| bwd_pkts_payload.tot | Cumulative payload bytes sent in backward direction                 |     0 |      1.36104e+07 |  3554.5  | 111857    |
| bwd_pkts_payload.avg | Mean payload byte size of backward packets                          |     0 |   1457.05        |   127.02 |    185.04 |
| fwd_init_window_size | Initial TCP window advertisement in forward direction (bytes)       |     0 |  65535           | 37103.9  |  31714.2  |
| bwd_init_window_size | Initial TCP window advertisement in backward direction (bytes)      |     0 |  65535           | 18740.3  |  19740.4  |