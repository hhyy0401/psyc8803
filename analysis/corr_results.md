# DDM x Resting EEG Correlation Analysis

## Overview

- **EEG Features**: ROI-based band power from `preprocess_v1.csv` (7 ROIs x 13 features = 91)
- **DDM Parameters**: Boundary separation (a) and drift rate (v)
- **Correction**: FDR (Benjamini-Hochberg) per DDM variable
- **N subjects**: 184

## ROI Definitions

| ROI | Channels |
|-----|----------|
| Frontal | E3, E6, E8, E9 |
| Posterior | E26, E34, E45, E35, E37, E38 |
| Central | E12, E13, E19, E20, E28, E29, E30, E31 |
| Left Temporal | E14, E15, E21, E22 |
| Right Temporal | E41, E42, E47, E48 |
| Parietal Extended | E39, E40, E43, E44 |
| Prefrontal | E1, E2, E4, E5, E7, E10 |

## EEG Features per ROI

- Absolute band power: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-40 Hz)
- Relative band power: each band / total power
- Theta/beta ratio
- Peak alpha frequency (PAF)
- Spectral entropy

## Method 1: Load / NoLoad Mean

### Load_a

FDR-significant: **5** / 91 features
Nominally significant (p < .05): **12** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | +0.4204 | 0.0000*** | 0.0000 | +0.2226 | 0.0024 | 0.0377 | 184 |
| right_temporal_abs_beta | +0.2751 | 0.0002*** | 0.0071 | +0.2451 | 0.0008 | 0.0377 | 184 |
| central_abs_gamma | +0.2467 | 0.0007*** | 0.0153 | +0.2184 | 0.0029 | 0.0377 | 184 |
| central_abs_beta | +0.2449 | 0.0008*** | 0.0153 | +0.2041 | 0.0054 | 0.0551 | 184 |
| right_temporal_rel_gamma | +0.2441 | 0.0008*** | 0.0153 | +0.0599 | 0.4193 | 0.6937 | 184 |
| frontal_abs_gamma | +0.2050 | 0.0052** | 0.0796 | +0.2184 | 0.0029 | 0.0377 | 184 |
| left_temporal_abs_beta | +0.1740 | 0.0182* | 0.2365 | +0.1850 | 0.0119 | 0.0987 | 184 |
| prefrontal_abs_beta | +0.1696 | 0.0213* | 0.2426 | +0.1891 | 0.0101 | 0.0923 | 184 |
| frontal_abs_beta | +0.1654 | 0.0248* | 0.2509 | +0.1782 | 0.0155 | 0.1086 | 184 |
| posterior_abs_beta | +0.1589 | 0.0312* | 0.2835 | +0.2208 | 0.0026 | 0.0377 | 184 |
| prefrontal_abs_gamma | +0.1544 | 0.0364* | 0.3007 | +0.2397 | 0.0010 | 0.0377 | 184 |
| left_temporal_abs_gamma | +0.1453 | 0.0491* | 0.3724 | +0.2099 | 0.0042 | 0.0482 | 184 |
| right_temporal_spectral_entropy | +0.1371 | 0.0635 | 0.4444 | +0.0733 | 0.3228 | 0.6222 | 184 |
| right_temporal_theta_beta_ratio | -0.1329 | 0.0722 | 0.4693 | -0.0684 | 0.3563 | 0.6236 | 184 |
| right_temporal_rel_beta | +0.1278 | 0.0840 | 0.5073 | +0.0758 | 0.3067 | 0.6202 | 184 |

### Load_v

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **11** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | -0.2954 | 0.0000*** | 0.0043 | -0.1563 | 0.0341 | 0.8840 | 184 |
| right_temporal_rel_gamma | -0.2215 | 0.0025** | 0.1146 | -0.0907 | 0.2209 | 0.8840 | 184 |
| right_temporal_theta_beta_ratio | +0.2073 | 0.0047** | 0.1440 | +0.0859 | 0.2461 | 0.8840 | 184 |
| central_abs_gamma | -0.1885 | 0.0104* | 0.2361 | -0.1215 | 0.1005 | 0.8840 | 184 |
| left_temporal_theta_beta_ratio | +0.1774 | 0.0160* | 0.2906 | +0.0883 | 0.2335 | 0.8840 | 184 |
| right_temporal_rel_theta | +0.1683 | 0.0224* | 0.3198 | +0.1222 | 0.0986 | 0.8840 | 184 |
| right_temporal_abs_beta | -0.1657 | 0.0246* | 0.3198 | -0.1555 | 0.0351 | 0.8840 | 184 |
| central_theta_beta_ratio | +0.1574 | 0.0329* | 0.3553 | +0.0197 | 0.7902 | 0.9225 | 184 |
| central_abs_beta | -0.1554 | 0.0351* | 0.3553 | -0.1183 | 0.1097 | 0.8840 | 184 |
| prefrontal_theta_beta_ratio | +0.1519 | 0.0395* | 0.3597 | +0.0602 | 0.4169 | 0.8840 | 184 |
| frontal_theta_beta_ratio | +0.1461 | 0.0478* | 0.3953 | +0.0614 | 0.4080 | 0.8840 | 184 |
| right_temporal_spectral_entropy | -0.1331 | 0.0716 | 0.5116 | -0.1072 | 0.1473 | 0.8840 | 184 |
| central_rel_gamma | -0.1292 | 0.0804 | 0.5116 | -0.0655 | 0.3768 | 0.8840 | 184 |
| left_temporal_spectral_entropy | -0.1260 | 0.0883 | 0.5116 | -0.0960 | 0.1950 | 0.8840 | 184 |
| right_temporal_rel_beta | -0.1225 | 0.0977 | 0.5116 | -0.0683 | 0.3570 | 0.8840 | 184 |

### NoLoad_a

FDR-significant: **5** / 91 features
Nominally significant (p < .05): **13** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | +0.4598 | 0.0000*** | 0.0000 | +0.2747 | 0.0002 | 0.0146 | 184 |
| central_abs_gamma | +0.3348 | 0.0000*** | 0.0002 | +0.1576 | 0.0327 | 0.3442 | 184 |
| right_temporal_rel_gamma | +0.3080 | 0.0000*** | 0.0006 | +0.1746 | 0.0178 | 0.2758 | 184 |
| right_temporal_abs_beta | +0.2579 | 0.0004*** | 0.0093 | +0.1740 | 0.0182 | 0.2758 | 184 |
| central_abs_beta | +0.2225 | 0.0024** | 0.0437 | +0.0646 | 0.3836 | 0.9237 | 184 |
| prefrontal_abs_gamma | +0.1932 | 0.0086** | 0.1304 | +0.1907 | 0.0095 | 0.2758 | 184 |
| central_rel_gamma | +0.1883 | 0.0105* | 0.1363 | +0.0787 | 0.2884 | 0.8633 | 184 |
| frontal_abs_gamma | +0.1759 | 0.0169* | 0.1928 | +0.1649 | 0.0253 | 0.3291 | 184 |
| right_temporal_spectral_entropy | +0.1725 | 0.0192* | 0.1942 | +0.1056 | 0.1538 | 0.7388 | 184 |
| right_temporal_rel_beta | +0.1684 | 0.0223* | 0.2031 | +0.1376 | 0.0625 | 0.5173 | 184 |
| left_temporal_abs_gamma | +0.1612 | 0.0288* | 0.2333 | +0.1774 | 0.0160 | 0.2758 | 184 |
| posterior_abs_gamma | +0.1593 | 0.0308* | 0.2333 | +0.2211 | 0.0026 | 0.1164 | 184 |
| posterior_abs_beta | +0.1556 | 0.0349* | 0.2444 | +0.1497 | 0.0425 | 0.3868 | 184 |
| prefrontal_abs_beta | +0.1435 | 0.0521 | 0.3384 | +0.0879 | 0.2355 | 0.8064 | 184 |
| posterior_abs_theta | +0.1369 | 0.0639 | 0.3874 | +0.0622 | 0.4017 | 0.9237 | 184 |

### NoLoad_v

FDR-significant: **2** / 91 features
Nominally significant (p < .05): **16** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_theta_beta_ratio | +0.2837 | 0.0001*** | 0.0063 | +0.1642 | 0.0260 | 0.2573 | 184 |
| left_temporal_theta_beta_ratio | +0.2772 | 0.0001*** | 0.0063 | +0.1966 | 0.0075 | 0.2573 | 184 |
| right_temporal_abs_gamma | -0.2240 | 0.0022** | 0.0659 | -0.1652 | 0.0250 | 0.2573 | 184 |
| central_theta_beta_ratio | +0.2176 | 0.0030** | 0.0659 | +0.0947 | 0.2009 | 0.5712 | 184 |
| right_temporal_rel_gamma | -0.2134 | 0.0036** | 0.0659 | -0.1366 | 0.0645 | 0.3369 | 184 |
| right_temporal_rel_theta | +0.2070 | 0.0048** | 0.0659 | +0.1714 | 0.0200 | 0.2573 | 184 |
| prefrontal_theta_beta_ratio | +0.2058 | 0.0051** | 0.0659 | +0.1444 | 0.0506 | 0.3287 | 184 |
| left_temporal_rel_theta | +0.1949 | 0.0080** | 0.0911 | +0.1565 | 0.0339 | 0.2573 | 184 |
| frontal_theta_beta_ratio | +0.1664 | 0.0240* | 0.2129 | +0.1648 | 0.0254 | 0.2573 | 184 |
| left_temporal_rel_beta | -0.1655 | 0.0247* | 0.2129 | -0.1903 | 0.0097 | 0.2573 | 184 |
| right_temporal_rel_beta | -0.1644 | 0.0257* | 0.2129 | -0.1387 | 0.0605 | 0.3369 | 184 |
| central_rel_gamma | -0.1571 | 0.0332* | 0.2516 | -0.1320 | 0.0740 | 0.3369 | 184 |
| central_abs_gamma | -0.1520 | 0.0395* | 0.2549 | -0.1630 | 0.0270 | 0.2573 | 184 |
| frontal_abs_gamma | -0.1495 | 0.0428* | 0.2549 | -0.0650 | 0.3810 | 0.7537 | 184 |
| right_temporal_spectral_entropy | -0.1488 | 0.0438* | 0.2549 | -0.1278 | 0.0839 | 0.3635 | 184 |

## Method 2: Condition-Paired Mean

### Accuracy_Max_a

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **11** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | +0.4167 | 0.0000*** | 0.0000 | +0.2554 | 0.0005 | 0.0426 | 184 |
| right_temporal_abs_beta | +0.2373 | 0.0012** | 0.0537 | +0.1909 | 0.0094 | 0.2860 | 184 |
| right_temporal_rel_gamma | +0.2266 | 0.0020** | 0.0602 | +0.1322 | 0.0737 | 0.5957 | 184 |
| central_abs_gamma | +0.2071 | 0.0048** | 0.1092 | +0.1830 | 0.0129 | 0.2937 | 184 |
| left_temporal_abs_gamma | +0.1920 | 0.0090** | 0.1640 | +0.1765 | 0.0165 | 0.3009 | 184 |
| frontal_abs_gamma | +0.1594 | 0.0307* | 0.3674 | +0.1021 | 0.1678 | 0.5996    | 184 |
| right_temporal_theta_beta_ratio | -0.1582 | 0.0319* | 0.3674 | -0.0989 | 0.1817 | 0.6122 | 184 |
| central_abs_beta | +0.1579 | 0.0323* | 0.3674 | +0.1042 | 0.1593 | 0.5996 | 184 |
| prefrontal_abs_gamma | +0.1486 | 0.0441* | 0.3974 | +0.1500 | 0.0421 | 0.4794 | 184 |
| posterior_abs_beta | +0.1483 | 0.0445* | 0.3974 | +0.1683 | 0.0224 | 0.3400 | 184 |
| left_temporal_theta_beta_ratio | -0.1460 | 0.0480* | 0.3974 | -0.0712 | 0.3369 | 0.6928 | 184 |
| right_temporal_rel_theta | -0.1420 | 0.0545 | 0.4090 | -0.1223 | 0.0982 | 0.5957 | 184 |
| left_temporal_rel_theta | -0.1388 | 0.0603 | 0.4090 | -0.1236 | 0.0946 | 0.5957 | 184 |
| left_temporal_abs_beta | +0.1374 | 0.0629 | 0.4090 | +0.0964 | 0.1928 | 0.6267 | 184 |
| prefrontal_rel_theta | -0.1265 | 0.0871 | 0.5184 | -0.1334 | 0.0709 | 0.5957 | 184 |

### Accuracy_Max_v

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **11** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | -0.2708 | 0.0002*** | 0.0182 | -0.1484 | 0.0443 | 0.6797 | 184 |
| right_temporal_theta_beta_ratio | +0.2345 | 0.0014** | 0.0617 | +0.1086 | 0.1423 | 0.8091 | 184 |
| left_temporal_theta_beta_ratio | +0.2257 | 0.0021** | 0.0628 | +0.1450 | 0.0495 | 0.6797 | 184 |
| right_temporal_rel_gamma | -0.2135 | 0.0036** | 0.0821 | -0.1099 | 0.1375 | 0.8091 | 184 |
| right_temporal_rel_theta | +0.1988 | 0.0068** | 0.1241 | +0.1735 | 0.0185 | 0.6797 | 184 | 
| prefrontal_theta_beta_ratio | +0.1935 | 0.0085** | 0.1288 | +0.1160 | 0.1167 | 0.8091 | 184 |
| frontal_theta_beta_ratio | +0.1845 | 0.0122* | 0.1508 | +0.1087 | 0.1418 | 0.8091 | 184 |
| left_temporal_rel_theta | +0.1823 | 0.0133* | 0.1508 | +0.1757 | 0.0171 | 0.6797 | 184 |
| central_theta_beta_ratio | +0.1783 | 0.0154* | 0.1560 | +0.0770 | 0.2986 | 0.9008 | 184 |
| prefrontal_rel_theta | +0.1741 | 0.0181* | 0.1648 | +0.1497 | 0.0425 | 0.6797 | 184 |
| central_abs_gamma | -0.1695 | 0.0215* | 0.1775 | -0.1391 | 0.0598 | 0.6797 | 184 |
| right_temporal_abs_beta | -0.1397 | 0.0586 | 0.4320 | -0.0930 | 0.2090 | 0.9008 | 184 |
| central_rel_theta | +0.1380 | 0.0617 | 0.4320 | +0.1186 | 0.1087 | 0.8091 | 184 |
| central_rel_gamma | -0.1316 | 0.0749 | 0.4865 | -0.1280 | 0.0834 | 0.7630 | 184 |
| frontal_abs_gamma | -0.1271 | 0.0856 | 0.5192 | -0.0568 | 0.4441 | 0.9008 | 184 |

### Accuracy_Mid_a

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **14** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | +0.2826 | 0.0001*** | 0.0093 | +0.1318 | 0.0744 | 0.5211 | 184 |
| right_temporal_rel_gamma | +0.1982 | 0.0070** | 0.2741 | +0.1466 | 0.0470 | 0.5211 | 184 |
| prefrontal_abs_gamma | +0.1803 | 0.0143* | 0.2741 | +0.1778 | 0.0158 | 0.5211 | 184 |
| posterior_rel_theta | +0.1796 | 0.0147* | 0.2741 | +0.1105 | 0.1355 | 0.5292 | 184 |
| frontal_abs_gamma | +0.1748 | 0.0176* | 0.2741 | +0.1432 | 0.0525 | 0.5211 | 184 |
| right_temporal_abs_beta | +0.1741 | 0.0181* | 0.2741 | +0.0942 | 0.2036 | 0.5614 | 184 |
| prefrontal_abs_beta | +0.1688 | 0.0220* | 0.2862 | +0.1041 | 0.1595 | 0.5495 | 184 |
| central_abs_gamma | +0.1631 | 0.0269* | 0.3003 | +0.1223 | 0.0980 | 0.5292 | 184 |
| parietal_extended_rel_theta | +0.1560 | 0.0345* | 0.3003 | +0.1031 | 0.1636 | 0.5495 | 184 |
| prefrontal_rel_gamma | +0.1551 | 0.0356* | 0.3003 | +0.1760 | 0.0169 | 0.5211 | 184 |
| posterior_abs_theta | +0.1540 | 0.0369* | 0.3003 | +0.0470 | 0.5263 | 0.8266 | 184 |
| frontal_spectral_entropy | +0.1477 | 0.0454* | 0.3003 | +0.1742 | 0.0180 | 0.5211 | 184 |
| frontal_rel_theta | +0.1477 | 0.0454* | 0.3003 | +0.0961 | 0.1945 | 0.5537 | 184 |
| frontal_rel_gamma | +0.1472 | 0.0462* | 0.3003 | +0.1498 | 0.0424 | 0.5211 | 184 |
| right_temporal_spectral_entropy | +0.1442 | 0.0508 | 0.3079 | +0.1338 | 0.0702 | 0.5211 | 184 |

### Accuracy_Mid_v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **5** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | -0.2032 | 0.0057** | 0.5163 | -0.0940 | 0.2046 | 0.7757 | 184 |
| right_temporal_theta_beta_ratio | +0.1734 | 0.0186* | 0.5872 | +0.0808 | 0.2754 | 0.8354 | 184 |
| right_temporal_rel_theta | +0.1716 | 0.0199* | 0.5872 | +0.1399 | 0.0583 | 0.6826 | 184 |
| left_temporal_theta_beta_ratio | +0.1608 | 0.0293* | 0.5872 | +0.0889 | 0.2301 | 0.7857 | 184 |
| left_temporal_rel_delta | +0.1527 | 0.0385* | 0.5872 | +0.1526 | 0.0386 | 0.6826 | 184 |
| left_temporal_spectral_entropy | -0.1430 | 0.0528 | 0.5872 | -0.1106 | 0.1350 | 0.6826 | 184 |
| right_temporal_rel_delta | +0.1367 | 0.0642 | 0.5872 | +0.1186 | 0.1088 | 0.6826 | 184 |
| right_temporal_abs_beta | -0.1355 | 0.0666 | 0.5872 | -0.1629 | 0.0272 | 0.6826 | 184 |
| right_temporal_rel_alpha | -0.1348 | 0.0682 | 0.5872 | -0.1558 | 0.0347 | 0.6826 | 184 |
| posterior_rel_delta | +0.1252 | 0.0904 | 0.5872 | +0.1206 | 0.1029 | 0.6826 | 184 |
| left_temporal_abs_gamma | -0.1251 | 0.0906 | 0.5872 | -0.1190 | 0.1075 | 0.6826 | 184 |
| left_temporal_rel_alpha | -0.1225 | 0.0976 | 0.5872 | -0.1330 | 0.0718 | 0.6826 | 184 |
| posterior_rel_alpha | -0.1210 | 0.1019 | 0.5872 | -0.1291 | 0.0808 | 0.6826 | 184 |
| left_temporal_rel_theta | +0.1202 | 0.1042 | 0.5872 | +0.0739 | 0.3191 | 0.8838 | 184 |
| central_theta_beta_ratio | +0.1195 | 0.1060 | 0.5872 | -0.0070 | 0.9248 | 0.9975 | 184 |

### Neutral_a

FDR-significant: **7** / 91 features
Nominally significant (p < .05): **15** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | +0.4124 | 0.0000*** | 0.0000 | +0.2467 | 0.0007 | 0.0331 | 184 |
| right_temporal_abs_beta | +0.2778 | 0.0001*** | 0.0061 | +0.2417 | 0.0009 | 0.0331 | 184 |
| central_abs_gamma | +0.2592 | 0.0004*** | 0.0116 | +0.1326 | 0.0729 | 0.3683 | 184 |
| central_abs_beta | +0.2490 | 0.0007*** | 0.0149 | +0.1619 | 0.0281 | 0.2560 | 184 |
| right_temporal_rel_gamma | +0.2443 | 0.0008*** | 0.0151 | +0.1215 | 0.1003 | 0.4564 | 184 |
| prefrontal_abs_beta | +0.2196 | 0.0027** | 0.0395 | +0.1984 | 0.0069 | 0.0902 | 184 |
| prefrontal_abs_gamma | +0.2173 | 0.0030** | 0.0395 | +0.2389 | 0.0011 | 0.0331 | 184 |
| left_temporal_abs_beta | +0.2048 | 0.0053** | 0.0602 | +0.1517 | 0.0399 | 0.2790 | 184 |
| left_temporal_abs_gamma | +0.1912 | 0.0093** | 0.0943 | +0.1535 | 0.0374 | 0.2790 | 184 |
| frontal_abs_gamma | +0.1826 | 0.0131* | 0.1191 | +0.2107 | 0.0041 | 0.0662 | 184 |
| posterior_abs_beta | +0.1800 | 0.0145* | 0.1199 | +0.2300 | 0.0017 | 0.0384 | 184 |
| right_temporal_rel_beta | +0.1704 | 0.0207* | 0.1572 | +0.1492 | 0.0432 | 0.2806 | 184 |
| right_temporal_spectral_entropy | +0.1600 | 0.0300* | 0.2103 | +0.1365 | 0.0647 | 0.3679 | 184 |
| frontal_abs_beta | +0.1557 | 0.0348* | 0.2195 | +0.1652 | 0.0251 | 0.2535 | 184 |
| right_temporal_theta_beta_ratio | -0.1546 | 0.0362* | 0.2195 | -0.1337 | 0.0703 | 0.3683 | 184 |

### Neutral_v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **11** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | -0.2380 | 0.0011** | 0.1039 | -0.2206 | 0.0026 | 0.1727 | 184 |
| central_abs_gamma | -0.2137 | 0.0036** | 0.1572 | -0.2012 | 0.0062 | 0.1873 | 184 |
| right_temporal_abs_beta | -0.2023 | 0.0059** | 0.1572 | -0.2124 | 0.0038 | 0.1727 | 184 |
| right_temporal_theta_beta_ratio | +0.1966 | 0.0075** | 0.1572 | +0.0910 | 0.2194 | 0.7545 | 184 |
| central_abs_beta | -0.1931 | 0.0086** | 0.1572 | -0.1699 | 0.0212 | 0.3210 | 184 |
| left_temporal_theta_beta_ratio | +0.1812 | 0.0138* | 0.2027 | +0.1032 | 0.1632 | 0.7545 | 184 |
| central_theta_beta_ratio | +0.1760 | 0.0169* | 0.2027 | +0.0485 | 0.5129 | 0.7779 | 184 |
| prefrontal_theta_beta_ratio | +0.1745 | 0.0178* | 0.2027 | +0.1065 | 0.1502 | 0.7545 | 184 |
| left_temporal_abs_beta | -0.1608 | 0.0292* | 0.2952 | -0.1806 | 0.0141 | 0.3210 | 184 |
| central_rel_gamma | -0.1577 | 0.0326* | 0.2963 | -0.0900 | 0.2245 | 0.7545 | 184 |
| right_temporal_rel_gamma | -0.1519 | 0.0395* | 0.3268 | -0.0834 | 0.2602 | 0.7545 | 184 |
| right_temporal_rel_theta | +0.1410 | 0.0563 | 0.3885 | +0.1372 | 0.0633 | 0.5318 | 184 |
| frontal_abs_beta | -0.1399 | 0.0583 | 0.3885 | -0.1505 | 0.0415 | 0.4378 | 184 |
| frontal_theta_beta_ratio | +0.1378 | 0.0621 | 0.3885 | +0.1238 | 0.0941 | 0.7137 | 184 |
| prefrontal_abs_beta | -0.1344 | 0.0689 | 0.3885 | -0.1367 | 0.0643 | 0.5318 | 184 |

### Speed_Max_a

FDR-significant: **6** / 91 features
Nominally significant (p < .05): **11** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | +0.4671 | 0.0000*** | 0.0000 | +0.2335 | 0.0014 | 0.0802 | 184 |
| central_abs_gamma | +0.4505 | 0.0000*** | 0.0000 | +0.2261 | 0.0020 | 0.0802 | 184 |
| right_temporal_rel_gamma | +0.3501 | 0.0000*** | 0.0000 | +0.0830 | 0.2626 | 0.6289 | 184 |
| central_abs_beta | +0.2967 | 0.0000*** | 0.0010 | +0.1703 | 0.0208 | 0.1951 | 184 |
| central_rel_gamma | +0.2832 | 0.0001*** | 0.0018 | +0.0328 | 0.6582 | 0.8695 | 184 |
| right_temporal_abs_beta | +0.2385 | 0.0011** | 0.0169 | +0.1992 | 0.0067 | 0.0802 | 184 |
| right_temporal_rel_beta | +0.1834 | 0.0127* | 0.1649 | +0.0710 | 0.3384 | 0.6303 | 184 |
| right_temporal_spectral_entropy | +0.1780 | 0.0156* | 0.1779 | +0.0849 | 0.2521 | 0.6218 | 184 |
| frontal_abs_gamma | +0.1591 | 0.0310* | 0.2933 | +0.2004 | 0.0064 | 0.0802 | 184 |
| central_spectral_entropy | +0.1558 | 0.0347* | 0.2933 | +0.0708 | 0.3394 | 0.6303 | 184 |
| central_rel_beta | +0.1552 | 0.0355* | 0.2933 | +0.0043 | 0.9540 | 0.9742 | 184 |
| left_temporal_peak_alpha_freq | -0.1221 | 0.0986 | 0.7149 | -0.0727 | 0.3267 | 0.6303 | 184 |
| frontal_abs_beta | +0.1209 | 0.1021 | 0.7149 | +0.1448 | 0.0498 | 0.3776 | 184 |
| frontal_abs_theta | +0.1154 | 0.1189 | 0.7558 | +0.0947 | 0.2011 | 0.6218 | 184 |
| left_temporal_theta_beta_ratio | -0.1089 | 0.1410 | 0.7558 | -0.1107 | 0.1346 | 0.6075 | 184 |

### Speed_Max_v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **6** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | -0.2078 | 0.0046** | 0.1860 | -0.1656 | 0.0247 | 0.4911 | 184 |
| right_temporal_rel_gamma | -0.2033 | 0.0056** | 0.1860 | -0.1631 | 0.0270 | 0.4911 | 184 |
| right_temporal_theta_beta_ratio | +0.2013 | 0.0061** | 0.1860 | +0.0744 | 0.3154 | 0.7570 | 184 |
| left_temporal_theta_beta_ratio | +0.1897 | 0.0099** | 0.2256 | +0.1130 | 0.1267 | 0.7570 | 184 |
| right_temporal_rel_theta | +0.1600 | 0.0300* | 0.5467 | +0.0985 | 0.1834 | 0.7570 | 184 |
| frontal_abs_gamma | -0.1514 | 0.0402* | 0.6093 | -0.0707 | 0.3401 | 0.7570 | 184 |
| central_theta_beta_ratio | +0.1332 | 0.0715 | 0.7445 | +0.0360 | 0.6272 | 0.8220 | 184 |
| left_temporal_rel_beta | -0.1281 | 0.0832 | 0.7445 | -0.1506 | 0.0413 | 0.5370 | 184 |
| frontal_rel_gamma | -0.1229 | 0.0965 | 0.7445 | -0.0979 | 0.1862 | 0.7570 | 184 |
| right_temporal_rel_beta | -0.1205 | 0.1032 | 0.7445 | -0.0817 | 0.2701 | 0.7570 | 184 |
| prefrontal_theta_beta_ratio | +0.1124 | 0.1287 | 0.7445 | +0.0496 | 0.5034 | 0.7570 | 184 |
| right_temporal_spectral_entropy | -0.1109 | 0.1341 | 0.7445 | -0.1022 | 0.1674 | 0.7570 | 184 |
| left_temporal_rel_theta | +0.1104 | 0.1356 | 0.7445 | +0.0677 | 0.3615 | 0.7570 | 184 |
| posterior_peak_alpha_freq | -0.1083 | 0.1434 | 0.7445 | -0.1014 | 0.1706 | 0.7570 | 184 |
| prefrontal_rel_theta | +0.1082 | 0.1437 | 0.7445 | +0.0898 | 0.2255 | 0.7570 | 184 |

### Speed_Mid_a

FDR-significant: **2** / 91 features
Nominally significant (p < .05): **8** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | +0.3332 | 0.0000*** | 0.0003 | +0.1901 | 0.0098 | 0.1481 | 184 |
| right_temporal_abs_beta | +0.2577 | 0.0004*** | 0.0188 | +0.2158 | 0.0033 | 0.1481 | 184 |
| central_abs_delta | +0.1862 | 0.0114* | 0.2894 | +0.1476 | 0.0456 | 0.3126 | 184 |
| posterior_abs_beta | +0.1778 | 0.0158* | 0.2894 | +0.1961 | 0.0076 | 0.1481 | 184 |
| central_abs_beta | +0.1775 | 0.0159* | 0.2894 | +0.1526 | 0.0386 | 0.3126 | 184 |
| frontal_abs_gamma | +0.1701 | 0.0210* | 0.3179 | +0.1585 | 0.0316 | 0.3061 | 184 |
| right_temporal_rel_gamma | +0.1475 | 0.0456* | 0.5118 | +0.0266 | 0.7200 | 0.9100 | 184 |
| parietal_extended_abs_beta | +0.1474 | 0.0458* | 0.5118 | +0.1933 | 0.0086 | 0.1481 | 184 |
| left_temporal_abs_beta | +0.1429 | 0.0530 | 0.5118 | +0.1354 | 0.0668 | 0.3577 | 184 |
| prefrontal_abs_delta | +0.1410 | 0.0562 | 0.5118 | +0.0478 | 0.5194 | 0.8990 | 184 |
| central_abs_gamma | +0.1293 | 0.0803 | 0.6019 | +0.1689 | 0.0219 | 0.2853 | 184 |
| posterior_abs_gamma | +0.1286 | 0.0819 | 0.6019 | +0.2092 | 0.0044 | 0.1481 | 184 |
| left_temporal_abs_delta | +0.1263 | 0.0876 | 0.6019 | +0.1438 | 0.0515 | 0.3126 | 184 |
| prefrontal_abs_gamma | +0.1223 | 0.0981 | 0.6019 | +0.1567 | 0.0336 | 0.3061 | 184 |
| posterior_abs_theta | +0.1219 | 0.0992 | 0.6019 | +0.1146 | 0.1212 | 0.4243 | 184 |

### Speed_Mid_v

FDR-significant: **6** / 91 features
Nominally significant (p < .05): **23** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_theta_beta_ratio | +0.2660 | 0.0003*** | 0.0169 | +0.1841 | 0.0123 | 0.1490 | 184 |
| right_temporal_rel_gamma | -0.2597 | 0.0004*** | 0.0169 | -0.1614 | 0.0286 | 0.2003 | 184 |
| right_temporal_spectral_entropy | -0.2473 | 0.0007*** | 0.0210 | -0.2225 | 0.0024 | 0.1246 | 184 |
| right_temporal_rel_beta | -0.2420 | 0.0009*** | 0.0210 | -0.2052 | 0.0052 | 0.1246 | 184 |
| left_temporal_theta_beta_ratio | +0.2378 | 0.0012** | 0.0210 | +0.1795 | 0.0147 | 0.1490 | 184 |
| central_theta_beta_ratio | +0.2164 | 0.0032** | 0.0482 | +0.1099 | 0.1375 | 0.3453 | 184 |
| right_temporal_abs_gamma | -0.2073 | 0.0048** | 0.0553 | -0.1266 | 0.0868 | 0.2724 | 184 |
| central_rel_gamma | -0.2068 | 0.0049** | 0.0553 | -0.0967 | 0.1915 | 0.4357 | 184 |
| central_rel_beta | -0.2007 | 0.0063** | 0.0637 | -0.1558 | 0.0347 | 0.2103 | 184 |
| prefrontal_theta_beta_ratio | +0.1956 | 0.0078** | 0.0710 | +0.1708 | 0.0204 | 0.1597 | 184 |
| central_abs_gamma | -0.1914 | 0.0092** | 0.0764 | -0.0680 | 0.3591 | 0.6052 | 184 |
| posterior_theta_beta_ratio | +0.1819 | 0.0134* | 0.1019 | +0.1322 | 0.0737 | 0.2532 | 184 |
| posterior_spectral_entropy | -0.1773 | 0.0161* | 0.1069 | -0.2175 | 0.0030 | 0.1246 | 184 |
| central_spectral_entropy | -0.1759 | 0.0169* | 0.1069 | -0.1509 | 0.0409 | 0.2324 | 184 |
| left_temporal_rel_beta | -0.1748 | 0.0176* | 0.1069 | -0.1875 | 0.0108 | 0.1490 | 184 |

## Method 3: Grand Mean

### a

FDR-significant: **5** / 91 features
Nominally significant (p < .05): **15** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | +0.4695 | 0.0000*** | 0.0000 | +0.2507 | 0.0006 | 0.0457 | 184 |
| central_abs_gamma | +0.3053 | 0.0000*** | 0.0011 | +0.2072 | 0.0048 | 0.0622 | 184 |
| right_temporal_rel_gamma | +0.2914 | 0.0001*** | 0.0017 | +0.1084 | 0.1429 | 0.6138 | 184 |
| right_temporal_abs_beta | +0.2875 | 0.0001*** | 0.0017 | +0.2206 | 0.0026 | 0.0597 | 184 |
| central_abs_beta | +0.2526 | 0.0005*** | 0.0098 | +0.1539 | 0.0370 | 0.3064 | 184 |
| frontal_abs_gamma | +0.2067 | 0.0049** | 0.0740 | +0.2071 | 0.0048 | 0.0622 | 184 |
| prefrontal_abs_gamma | +0.1836 | 0.0126* | 0.1639 | +0.2340 | 0.0014 | 0.0457 | 184 |
| prefrontal_abs_beta | +0.1701 | 0.0210* | 0.2125 | +0.1507 | 0.0411 | 0.3120 | 184 |
| posterior_abs_beta | +0.1691 | 0.0217* | 0.2125 | +0.1968 | 0.0074 | 0.0844 | 184 |
| right_temporal_spectral_entropy | +0.1634 | 0.0266* | 0.2125 | +0.0844 | 0.2548 | 0.6138 | 184 |
| left_temporal_abs_gamma | +0.1633 | 0.0268* | 0.2125 | +0.2118 | 0.0039 | 0.0622 | 184 |
| left_temporal_abs_beta | +0.1599 | 0.0302* | 0.2125 | +0.1458 | 0.0482 | 0.3375 | 184 |
| central_rel_gamma | +0.1597 | 0.0304* | 0.2125 | +0.0570 | 0.4425 | 0.6647 | 184 |
| right_temporal_rel_beta | +0.1558 | 0.0347* | 0.2254 | +0.0945 | 0.2022 | 0.6138 | 184 |
| frontal_abs_beta | +0.1500 | 0.0422* | 0.2558 | +0.1408 | 0.0566 | 0.3676 | 184 |

### v

FDR-significant: **4** / 91 features
Nominally significant (p < .05): **16** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| right_temporal_abs_gamma | -0.2835 | 0.0001*** | 0.0088 | -0.1766 | 0.0165 | 0.3404 | 184 |
| right_temporal_theta_beta_ratio | +0.2711 | 0.0002*** | 0.0090 | +0.1411 | 0.0561 | 0.4612 | 184 |
| left_temporal_theta_beta_ratio | +0.2516 | 0.0006*** | 0.0173 | +0.1650 | 0.0252 | 0.3404 | 184 |
| right_temporal_rel_gamma | -0.2384 | 0.0011** | 0.0254 | -0.1243 | 0.0927 | 0.4684 | 184 |
| central_theta_beta_ratio | +0.2070 | 0.0048** | 0.0737 | +0.0715 | 0.3347 | 0.7083 | 184 |
| right_temporal_rel_theta | +0.2068 | 0.0049** | 0.0737 | +0.1663 | 0.0240 | 0.3404 | 184 |
| prefrontal_theta_beta_ratio | +0.1975 | 0.0072** | 0.0938 | +0.1203 | 0.1037 | 0.4967 | 184 |
| central_abs_gamma | -0.1860 | 0.0115* | 0.1304 | -0.1621 | 0.0279 | 0.3404 | 184 |
| frontal_theta_beta_ratio | +0.1719 | 0.0196* | 0.1747 | +0.1336 | 0.0706 | 0.4612 | 184 |
| right_temporal_abs_beta | -0.1718 | 0.0197* | 0.1747 | -0.1686 | 0.0222 | 0.3404 | 184 |
| left_temporal_rel_theta | +0.1699 | 0.0211* | 0.1747 | +0.1311 | 0.0760 | 0.4612 | 184 |
| right_temporal_rel_beta | -0.1583 | 0.0318* | 0.2226 | -0.1092 | 0.1400 | 0.5774 | 184 |
| central_rel_gamma | -0.1577 | 0.0325* | 0.2226 | -0.1093 | 0.1398 | 0.5774 | 184 |
| right_temporal_spectral_entropy | -0.1551 | 0.0356* | 0.2226 | -0.1343 | 0.0691 | 0.4612 | 184 |
| left_temporal_rel_beta | -0.1542 | 0.0367* | 0.2226 | -0.1662 | 0.0241 | 0.3404 | 184 |

## Summary: FDR-Significant Results

**46** correlations survived FDR correction:

| Method | DDM Var | EEG Feature | r | p | FDR p |
|--------|---------|-------------|---|---|-------|
| GrandMean | a | right_temporal_abs_gamma | +0.4695 | 0.0000 | 0.0000 |
| PairedMean | Speed_Max_a | right_temporal_abs_gamma | +0.4671 | 0.0000 | 0.0000 |
| Load/NoLoad | NoLoad_a | right_temporal_abs_gamma | +0.4598 | 0.0000 | 0.0000 |
| PairedMean | Speed_Max_a | central_abs_gamma | +0.4505 | 0.0000 | 0.0000 |
| Load/NoLoad | Load_a | right_temporal_abs_gamma | +0.4204 | 0.0000 | 0.0000 |
| PairedMean | Accuracy_Max_a | right_temporal_abs_gamma | +0.4167 | 0.0000 | 0.0000 |
| PairedMean | Neutral_a | right_temporal_abs_gamma | +0.4124 | 0.0000 | 0.0000 |
| PairedMean | Speed_Max_a | right_temporal_rel_gamma | +0.3501 | 0.0000 | 0.0000 |
| Load/NoLoad | NoLoad_a | central_abs_gamma | +0.3348 | 0.0000 | 0.0002 |
| PairedMean | Speed_Mid_a | right_temporal_abs_gamma | +0.3332 | 0.0000 | 0.0003 |
| Load/NoLoad | NoLoad_a | right_temporal_rel_gamma | +0.3080 | 0.0000 | 0.0006 |
| PairedMean | Speed_Max_a | central_abs_beta | +0.2967 | 0.0000 | 0.0010 |
| GrandMean | a | central_abs_gamma | +0.3053 | 0.0000 | 0.0011 |
| GrandMean | a | right_temporal_abs_beta | +0.2875 | 0.0001 | 0.0017 |
| GrandMean | a | right_temporal_rel_gamma | +0.2914 | 0.0001 | 0.0017 |
| PairedMean | Speed_Max_a | central_rel_gamma | +0.2832 | 0.0001 | 0.0018 |
| Load/NoLoad | Load_v | right_temporal_abs_gamma | -0.2954 | 0.0000 | 0.0043 |
| PairedMean | Neutral_a | right_temporal_abs_beta | +0.2778 | 0.0001 | 0.0061 |
| Load/NoLoad | NoLoad_v | left_temporal_theta_beta_ratio | +0.2772 | 0.0001 | 0.0063 |
| Load/NoLoad | NoLoad_v | right_temporal_theta_beta_ratio | +0.2837 | 0.0001 | 0.0063 |
| Load/NoLoad | Load_a | right_temporal_abs_beta | +0.2751 | 0.0002 | 0.0071 |
| GrandMean | v | right_temporal_abs_gamma | -0.2835 | 0.0001 | 0.0088 |
| GrandMean | v | right_temporal_theta_beta_ratio | +0.2711 | 0.0002 | 0.0090 |
| PairedMean | Accuracy_Mid_a | right_temporal_abs_gamma | +0.2826 | 0.0001 | 0.0093 |
| Load/NoLoad | NoLoad_a | right_temporal_abs_beta | +0.2579 | 0.0004 | 0.0093 |
| GrandMean | a | central_abs_beta | +0.2526 | 0.0005 | 0.0098 |
| PairedMean | Neutral_a | central_abs_gamma | +0.2592 | 0.0004 | 0.0116 |
| PairedMean | Neutral_a | central_abs_beta | +0.2490 | 0.0007 | 0.0149 |
| PairedMean | Neutral_a | right_temporal_rel_gamma | +0.2443 | 0.0008 | 0.0151 |
| Load/NoLoad | Load_a | central_abs_beta | +0.2449 | 0.0008 | 0.0153 |
| Load/NoLoad | Load_a | right_temporal_rel_gamma | +0.2441 | 0.0008 | 0.0153 |
| Load/NoLoad | Load_a | central_abs_gamma | +0.2467 | 0.0007 | 0.0153 |
| PairedMean | Speed_Mid_v | right_temporal_rel_gamma | -0.2597 | 0.0004 | 0.0169 |
| PairedMean | Speed_Mid_v | right_temporal_theta_beta_ratio | +0.2660 | 0.0003 | 0.0169 |
| PairedMean | Speed_Max_a | right_temporal_abs_beta | +0.2385 | 0.0011 | 0.0169 |
| GrandMean | v | left_temporal_theta_beta_ratio | +0.2516 | 0.0006 | 0.0173 |
| PairedMean | Accuracy_Max_v | right_temporal_abs_gamma | -0.2708 | 0.0002 | 0.0182 |
| PairedMean | Speed_Mid_a | right_temporal_abs_beta | +0.2577 | 0.0004 | 0.0188 |
| PairedMean | Speed_Mid_v | left_temporal_theta_beta_ratio | +0.2378 | 0.0012 | 0.0210 |
| PairedMean | Speed_Mid_v | right_temporal_rel_beta | -0.2420 | 0.0009 | 0.0210 |
| PairedMean | Speed_Mid_v | right_temporal_spectral_entropy | -0.2473 | 0.0007 | 0.0210 |
| GrandMean | v | right_temporal_rel_gamma | -0.2384 | 0.0011 | 0.0254 |
| PairedMean | Neutral_a | prefrontal_abs_beta | +0.2196 | 0.0027 | 0.0395 |
| PairedMean | Neutral_a | prefrontal_abs_gamma | +0.2173 | 0.0030 | 0.0395 |
| Load/NoLoad | NoLoad_a | central_abs_beta | +0.2225 | 0.0024 | 0.0437 |
| PairedMean | Speed_Mid_v | central_theta_beta_ratio | +0.2164 | 0.0032 | 0.0482 |
