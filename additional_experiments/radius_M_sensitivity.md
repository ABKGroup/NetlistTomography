# Sensitivity of the Co-location Radius to Sample Size M

The adaptive co-location radius is estimated as
$r_{adapt} = q \cdot \mathrm{median}(\text{1-NN distances})$, where "1-NN
distance" is a sampled cell's nearest-neighbor distance. The sampled cells are
used only to estimate this radius; they are **not** a subsample of cells used
for training or clustering.

Computing 1-NN distances over all instances can be expensive, especially for
large designs. Therefore we extract a subset of $M$ cells ($M = 10K$ in the
paper) to estimate the radius. This study shows that the radius estimate is
stable with respect to the choice of $M$.

We conduct a sensitivity analysis by sweeping
$M \in \{1\text{K}, 5\text{K}, 10\text{K}, 20\text{K}, 30\text{K}\}$ across
all ten designs. As shown in the table below, the estimated radius changes by
at most 0.61% across all these designs. The estimate converges before 10K for
most designs. We therefore use 10K as a conservative default: it ensures the
radius estimation is stable, while avoiding unnecessary nearest-neighbor
computation.

The full sweep is provided in
[`data/radius_M_summary.csv`](data/radius_M_summary.csv).

| Design | $r_{\mathrm{adapt}}$ @1K | @5K | @10K | @20K | @30K | Max $\Delta r_{\mathrm{adapt}}$ |
|---|---:|---:|---:|---:|---:|---:|
| JPEG_NG45    | 2.271 | 2.280 | 2.280 | 2.280 | 2.280 | 0.39% |
| SweRV_NG45   | 2.936 | 2.929 | 2.926 | 2.926 | 2.926 | 0.34% |
| Ariane_NG45  | 3.821 | 3.833 | 3.840 | 3.840 | 3.840 | 0.49% |
| CA53_NG45    | 4.474 | 4.491 | 4.495 | 4.496 | 4.496 | 0.48% |
| BP_Quad_NG45 | 9.125 | 9.122 | 9.119 | 9.118 | 9.118 | 0.07% |
| JPEG_GF12    | 1.355 | 1.357 | 1.356 | 1.356 | 1.356 | 0.07% |
| SweRV_GF12   | 1.795 | 1.788 | 1.788 | 1.788 | 1.788 | 0.39% |
| Ariane_GF12  | 1.767 | 1.766 | 1.764 | 1.764 | 1.764 | 0.17% |
| CA53_GF12    | 2.713 | 2.717 | 2.718 | 2.718 | 2.718 | 0.18% |
| BP_Quad_GF12 | 3.925 | 3.938 | 3.948 | 3.949 | 3.949 | 0.61% |
