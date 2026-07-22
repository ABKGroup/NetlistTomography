# TomoGNN Component Ablation

This study isolates the individual contributions of three TomoGNN components:
FiLM modulation, dual-graph message passing, and the tomography calibration
loss.

We perform **one-component-removal ablations** on JPEG (NG45, GF12) and CA53
(NG45, GF12). Specifically, we compare the full TomoGNN model against variants
without FiLM modulation, without dual-graph message passing, and without the
tomography calibration loss. The variant without dual-graph message passing
uses three-layer GraphSAGE only on the netlist graph. All other settings are
kept identical, and we report post-routeOpt rWL, power, WNS, TNS, DRV, and
runtime.

These ablations show that all three components contribute to final timing
quality. Removing FiLM and dual-graph MP worsens TNS in all four cases. Removing
the calibration loss worsens TNS in 3/4 cases. In the remaining CA53 GF12 case,
it slightly improves TNS but increases DRV and runtime, indicating a worse
overall tradeoff. Overall, the full TomoGNN gives the most consistent post-route
timing improvement across designs and PDKs.

The underlying numbers are provided in
[`data/tomognn_ablation.csv`](data/tomognn_ablation.csv).


| Design | Variant | rWL | Power | WNS | TNS | DRV | RT |
|---|---|---:|---:|---:|---:|---:|---:|
| **JPEG_NG45** | **Full TomoGNN**| **498** | **317** | **-119** | **-70.8** | **0** | **0.63** |
| JPEG_NG45 | w/o FiLM | 497 | 315 | -112 | -75.3 | 0 | 0.69 |
| JPEG_NG45 | w/o dual-graph MP | 497 | 317 | -158 | -81.0 | 0 | 0.71 |
| JPEG_NG45 | w/o calibration loss | 498 | 317 | -145 | -81.9 | 0 | 0.58 |
| **CA53_NG45** | **Full TomoGNN** | **11139** | **844** | **-53** | **-56.2** | **1057** | **3.97** |
| CA53_NG45 | w/o FiLM | 11138 | 844 | -59 | -95.5 | 1392 | 4.01 |
| CA53_NG45 | w/o dual-graph MP | 11139 | 844 | -57 | -69.5 | 983 | 4.00 |
| CA53_NG45 | w/o calibration loss | 11109 | 843 | -61 | -87.3 | 1129 | 3.74 |
| **JPEG_GF12** | **Full TomoGNN** | **0.948** | **0.980** | **-0.325** | **-335** | **0** | **1.46** |
| JPEG_GF12 | w/o FiLM | 0.940 | 0.988 | -0.320 | -397 | 0 | 1.57 |
| JPEG_GF12 | w/o dual-graph MP | 0.963 | 0.996 | -0.430 | -413 | 0 | 1.45 |
| JPEG_GF12 | w/o calibration loss | 0.929 | 0.986 | -0.485 | -488 | 0 | 1.49 |
| **CA53_GF12** | **Full TomoGNN** | **1.000** | **0.999** |**-0.183** | **-162** | **1** | **4.88** |
| CA53_GF12 | w/o FiLM | 0.999 | 1.000 | -0.223 | -167 | 2 | 6.48 |
| CA53_GF12 | w/o dual-graph MP | 1.001 | 1.000 | -0.182 | -176 | 1 | 5.06 |
| CA53_GF12 | w/o calibration loss | 1.001 | 1.000 | -0.175 | -160 | 2 | 6.13 |

*rWL/Power for JPEG_GF12 and CA53_GF12 are reported normalized to the full-model
baseline; NG45 values are absolute.*
