# DDM x Resting EEG Correlation Analysis

## Overview

- **EEG Features**: ROI-based band power from `preprocess_v2.csv` (7 ROIs x 13 features = 91)
- **DDM Parameters**: Boundary separation (a) and drift rate (v)
- **Correction**: FDR (Benjamini-Hochberg) per DDM variable
- **N subjects**: 184

## ROI Definitions

| ROI | Channels |
|-----|----------|
| Frontal | E3, E6, E8, E9 |
| Posterior | E34, E31, E40, E33, E38, E36 |
| Central | E16, E7, E4, E54, E51, E41, E21 |
| Left Temporal | E22, E24, E25, E30 |
| Right Temporal | E52, E48, E45, E44 |
| Occipital | E36, E37, E39 |
| Prefrontal | E1, E17, E2, E11, E5, E10 |

## EEG Features per ROI

- Absolute band power: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-40 Hz)
- Relative band power: each band / total power
- Theta/beta ratio
- Peak alpha frequency (PAF)
- Spectral entropy

## Method 1: Load / NoLoad Mean

### Load_a

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **8** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_abs_gamma | +0.2377 | 0.0012** | 0.1053 | +0.2318 | 0.0015 | 0.0659 | 184 |
| left_temporal_abs_gamma | +0.2219 | 0.0025** | 0.1122 | +0.2209 | 0.0026 | 0.0659 | 184 |
| left_temporal_abs_beta | +0.2104 | 0.0041** | 0.1194 | +0.1913 | 0.0093 | 0.1127 | 184 |
| frontal_abs_gamma | +0.2050 | 0.0052** | 0.1194 | +0.2184 | 0.0029 | 0.0659 | 184 |
| central_abs_beta | +0.1948 | 0.0081** | 0.1465 | +0.1833 | 0.0128 | 0.1127 | 184 |
| left_temporal_rel_gamma | +0.1733 | 0.0187* | 0.2829 | +0.0606 | 0.4142 | 0.6613 | 184 |
| frontal_abs_beta | +0.1654 | 0.0248* | 0.3151 | +0.1782 | 0.0155 | 0.1127 | 184 |
| right_temporal_abs_beta | +0.1623 | 0.0277* | 0.3151 | +0.1849 | 0.0120 | 0.1127 | 184 |
| prefrontal_abs_beta | +0.1403 | 0.0576 | 0.5409 | +0.1772 | 0.0161 | 0.1127 | 184 |
| left_temporal_spectral_entropy | +0.1392 | 0.0594 | 0.5409 | +0.0937 | 0.2060 | 0.5189 | 184 |
| prefrontal_abs_gamma | +0.1323 | 0.0735 | 0.5911 | +0.2270 | 0.0019 | 0.0659 | 184 |
| central_spectral_entropy | +0.1277 | 0.0841 | 0.5911 | +0.1102 | 0.1364 | 0.4596 | 184 |
| right_temporal_abs_gamma | +0.1237 | 0.0942 | 0.5911 | +0.1949 | 0.0080 | 0.1127 | 184 |
| posterior_abs_beta | +0.1233 | 0.0955 | 0.5911 | +0.1910 | 0.0094 | 0.1127 | 184 |
| central_rel_gamma | +0.1226 | 0.0974 | 0.5911 | +0.0911 | 0.2186 | 0.5234 | 184 |

### Load_v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **8** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_theta_beta_ratio | +0.1846 | 0.0121* | 0.5348 | +0.1083 | 0.1433 | 0.8680 | 184 |
| left_temporal_theta_beta_ratio | +0.1674 | 0.0231* | 0.5348 | +0.0577 | 0.4362 | 0.8680 | 184 |
| central_spectral_entropy | -0.1594 | 0.0307* | 0.5348 | -0.1432 | 0.0525 | 0.8680 | 184 |
| left_temporal_rel_gamma | -0.1591 | 0.0310* | 0.5348 | -0.0734 | 0.3224 | 0.8680 | 184 |
| central_abs_gamma | -0.1516 | 0.0399* | 0.5348 | -0.1527 | 0.0385 | 0.8680 | 184 |
| left_temporal_rel_theta | +0.1497 | 0.0425* | 0.5348 | +0.1102 | 0.1365 | 0.8680 | 184 |
| left_temporal_abs_gamma | -0.1487 | 0.0439* | 0.5348 | -0.1728 | 0.0190 | 0.8680 | 184 |
| frontal_theta_beta_ratio | +0.1461 | 0.0478* | 0.5348 | +0.0614 | 0.4080 | 0.8680 | 184 |
| right_temporal_theta_beta_ratio | +0.1429 | 0.0529 | 0.5348 | +0.0320 | 0.6664 | 0.8993 | 184 |
| right_temporal_rel_theta | +0.1325 | 0.0730 | 0.6199 | +0.0759 | 0.3055 | 0.8680 | 184 |
| left_temporal_abs_beta | -0.1295 | 0.0797 | 0.6199 | -0.1404 | 0.0573 | 0.8680 | 184 |
| central_rel_gamma | -0.1257 | 0.0891 | 0.6199 | -0.1434 | 0.0522 | 0.8680 | 184 |
| central_abs_beta | -0.1247 | 0.0916 | 0.6199 | -0.1137 | 0.1243 | 0.8680 | 184 |
| frontal_abs_delta | +0.1223 | 0.0981 | 0.6199 | +0.0257 | 0.7286 | 0.9339 | 184 |
| central_rel_delta | +0.1176 | 0.1117 | 0.6199 | +0.0857 | 0.2475 | 0.8680 | 184 |

### NoLoad_a

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **11** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| left_temporal_abs_gamma | +0.2882 | 0.0001*** | 0.0066 | +0.1361 | 0.0655 | 0.4800 | 184 |
| right_temporal_abs_gamma | +0.2229 | 0.0024** | 0.1072 | +0.2340 | 0.0014 | 0.0825 | 184 |
| left_temporal_rel_gamma | +0.2096 | 0.0043** | 0.1180 | +0.1130 | 0.1267 | 0.7215 | 184 |
| right_temporal_abs_beta | +0.2053 | 0.0052** | 0.1180 | +0.1346 | 0.0686 | 0.4800 | 184 |
| right_temporal_rel_gamma | +0.1961 | 0.0076** | 0.1388 | +0.1757 | 0.0170 | 0.3102 | 184 |
| left_temporal_abs_beta | +0.1913 | 0.0093** | 0.1406 | +0.0518 | 0.4847 | 0.9311 | 184 |
| central_abs_gamma | +0.1825 | 0.0132* | 0.1710 | +0.2096 | 0.0043 | 0.1301 | 184 |
| frontal_abs_gamma | +0.1759 | 0.0169* | 0.1928 | +0.1649 | 0.0253 | 0.3572 | 184 |
| prefrontal_abs_gamma | +0.1708 | 0.0205* | 0.2069 | +0.1802 | 0.0144 | 0.3102 | 184 |
| prefrontal_abs_delta | +0.1500 | 0.0421* | 0.3753 | +0.0799 | 0.2809 | 0.8355 | 184 |
| posterior_abs_beta | +0.1477 | 0.0454* | 0.3753 | +0.1626 | 0.0275 | 0.3572 | 184 |
| posterior_abs_theta | +0.1389 | 0.0601 | 0.4560 | +0.1033 | 0.1628 | 0.7215 | 184 |
| occipital_abs_theta | +0.1320 | 0.0741 | 0.4829 | +0.0420 | 0.5716 | 0.9311 | 184 |
| right_temporal_rel_beta | +0.1319 | 0.0743 | 0.4829 | +0.1101 | 0.1369 | 0.7215 | 184 |
| posterior_abs_gamma | +0.1264 | 0.0873 | 0.5255 | +0.2285 | 0.0018 | 0.0825 | 184 |

### NoLoad_v

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **16** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_theta_beta_ratio | +0.2705 | 0.0002*** | 0.0186 | +0.2334 | 0.0014 | 0.1299 | 184 |
| right_temporal_theta_beta_ratio | +0.2261 | 0.0020** | 0.0921 | +0.1254 | 0.0899 | 0.5114 | 184 |
| left_temporal_theta_beta_ratio | +0.2088 | 0.0044** | 0.1347 | +0.0864 | 0.2433 | 0.6919 | 184 |
| left_temporal_rel_gamma | -0.1876 | 0.0108* | 0.2038 | -0.1197 | 0.1055 | 0.5647 | 184 |
| central_rel_beta | -0.1821 | 0.0134* | 0.2038 | -0.2093 | 0.0043 | 0.1424 | 184 |
| left_temporal_abs_gamma | -0.1818 | 0.0135* | 0.2038 | -0.1565 | 0.0339 | 0.3451 | 184 |
| central_rel_gamma | -0.1779 | 0.0157* | 0.2038 | -0.1815 | 0.0137 | 0.2936 | 184 |
| prefrontal_theta_beta_ratio | +0.1737 | 0.0184* | 0.2093 | +0.0985 | 0.1835 | 0.5927 | 184 |
| left_temporal_rel_theta | +0.1698 | 0.0212* | 0.2141 | +0.1165 | 0.1153 | 0.5828 | 184 |
| frontal_theta_beta_ratio | +0.1664 | 0.0240* | 0.2181 | +0.1648 | 0.0254 | 0.3451 | 184 |
| central_abs_gamma | -0.1592 | 0.0309* | 0.2365 | -0.2076 | 0.0047 | 0.1424 | 184 |
| central_abs_beta | -0.1552 | 0.0353* | 0.2365 | -0.1772 | 0.0161 | 0.2936 | 184 |
| central_rel_theta | +0.1549 | 0.0357* | 0.2365 | +0.1532 | 0.0379 | 0.3451 | 184 |
| right_temporal_rel_theta | +0.1544 | 0.0364* | 0.2365 | +0.1044 | 0.1584 | 0.5856 | 184 |
| prefrontal_rel_theta | +0.1509 | 0.0408* | 0.2435 | +0.0867 | 0.2417 | 0.6919 | 184 |

## Method 2: Condition-Paired Mean

### Accuracy_Max_a

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **13** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| left_temporal_abs_gamma | +0.2848 | 0.0001*** | 0.0081 | +0.1980 | 0.0070 | 0.1786 | 184 |
| right_temporal_abs_gamma | +0.2148 | 0.0034** | 0.1054 | +0.2491 | 0.0007 | 0.0592 | 184 |
| left_temporal_rel_gamma | +0.2118 | 0.0039** | 0.1054 | +0.1532 | 0.0379 | 0.2619 | 184 |
| right_temporal_abs_beta | +0.2079 | 0.0046** | 0.1054 | +0.1899 | 0.0098 | 0.1786 | 184 |
| left_temporal_abs_beta | +0.1867 | 0.0112* | 0.1699 | +0.1044 | 0.1586 | 0.5120 | 184 |
| right_temporal_theta_beta_ratio | -0.1866 | 0.0112* | 0.1699 | -0.1753 | 0.0173 | 0.2162 | 184 |
| central_abs_gamma | +0.1770 | 0.0162* | 0.2109 | +0.2001 | 0.0065 | 0.1786 | 184 |
| right_temporal_rel_theta | -0.1658 | 0.0245* | 0.2504 | -0.1583 | 0.0318 | 0.2428 | 184 |
| right_temporal_rel_gamma | +0.1644 | 0.0257* | 0.2504 | +0.1724 | 0.0193 | 0.2162 | 184 |
| posterior_abs_beta | +0.1625 | 0.0275* | 0.2504 | +0.1696 | 0.0214 | 0.2162 | 184 |
| frontal_abs_gamma | +0.1594 | 0.0307* | 0.2540 | +0.1021 | 0.1678 | 0.5120 | 184 |
| prefrontal_rel_theta | -0.1537 | 0.0372* | 0.2821 | -0.1513 | 0.0403 | 0.2619 | 184 |
| prefrontal_abs_delta | +0.1465 | 0.0472* | 0.3302 | +0.1932 | 0.0086 | 0.1786 | 184 |
| left_temporal_theta_beta_ratio | -0.1425 | 0.0536 | 0.3483 | -0.1132 | 0.1261 | 0.5120 | 184 |
| prefrontal_abs_gamma | +0.1374 | 0.0629 | 0.3818 | +0.1594 | 0.0307 | 0.2428 | 184 |

### Accuracy_Max_v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **13** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_theta_beta_ratio | +0.2273 | 0.0019** | 0.1548 | +0.1643 | 0.0258 | 0.7833 | 184 |
| right_temporal_theta_beta_ratio | +0.2057 | 0.0051** | 0.1548 | +0.0851 | 0.2507 | 0.9756 | 184 |
| central_abs_gamma | -0.2056 | 0.0051** | 0.1548 | -0.1994 | 0.0067 | 0.6061 | 184 |
| frontal_theta_beta_ratio | +0.1845 | 0.0122* | 0.2589 | +0.1087 | 0.1418 | 0.8348 | 184 |
| central_rel_gamma | -0.1770 | 0.0162* | 0.2589 | -0.1669 | 0.0235 | 0.7833 | 184 |
| prefrontal_rel_theta | +0.1728 | 0.0190* | 0.2589 | +0.1419 | 0.0546 | 0.8288 | 184 |
| left_temporal_theta_beta_ratio | +0.1668 | 0.0236* | 0.2589 | +0.0861 | 0.2452 | 0.9756 | 184 |
| left_temporal_abs_gamma | -0.1616 | 0.0284* | 0.2589 | -0.1237 | 0.0942 | 0.8348 | 184 |
| central_rel_theta | +0.1613 | 0.0287* | 0.2589 | +0.1533 | 0.0378 | 0.8288 | 184 |
| left_temporal_rel_gamma | -0.1604 | 0.0297* | 0.2589 | -0.1188 | 0.1082 | 0.8348 | 184 |
| right_temporal_rel_theta | +0.1566 | 0.0338* | 0.2589 | +0.1325 | 0.0730 | 0.8348 | 184 |
| left_temporal_rel_theta | +0.1563 | 0.0341* | 0.2589 | +0.1446 | 0.0502 | 0.8288 | 184 |
| prefrontal_theta_beta_ratio | +0.1527 | 0.0385* | 0.2694 | +0.0798 | 0.2817 | 0.9756 | 184 |
| central_abs_beta | -0.1389 | 0.0600 | 0.3903 | -0.1050 | 0.1559 | 0.8348 | 184 |
| frontal_abs_gamma | -0.1271 | 0.0856 | 0.4696 | -0.0568 | 0.4441 | 0.9756 | 184 |

### Accuracy_Mid_a

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **12** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_abs_gamma | +0.1970 | 0.0073** | 0.3340 | +0.2075 | 0.0047 | 0.3709 | 184 |
| occipital_rel_theta | +0.1894 | 0.0100* | 0.3340 | +0.1197 | 0.1055 | 0.6860 | 184 |
| posterior_rel_theta | +0.1870 | 0.0110* | 0.3340 | +0.1021 | 0.1678 | 0.6871 | 184 |
| frontal_abs_gamma | +0.1748 | 0.0176* | 0.3340 | +0.1432 | 0.0525 | 0.6568 | 184 |
| left_temporal_abs_gamma | +0.1720 | 0.0196* | 0.3340 | +0.0883 | 0.2334 | 0.6871 | 184 |
| central_abs_beta | +0.1558 | 0.0346* | 0.3340 | +0.1002 | 0.1761 | 0.6871 | 184 |
| occipital_abs_theta | +0.1546 | 0.0361* | 0.3340 | +0.0335 | 0.6514 | 0.8596 | 184 |
| prefrontal_abs_gamma | +0.1543 | 0.0365* | 0.3340 | +0.1527 | 0.0385 | 0.6568 | 184 |
| left_temporal_abs_beta | +0.1523 | 0.0391* | 0.3340 | +0.0390 | 0.5990 | 0.8499 | 184 |
| frontal_spectral_entropy | +0.1477 | 0.0454* | 0.3340 | +0.1742 | 0.0180 | 0.5467 | 184 |
| frontal_rel_theta | +0.1477 | 0.0454* | 0.3340 | +0.0961 | 0.1945 | 0.6871 | 184 |
| frontal_rel_gamma | +0.1472 | 0.0462* | 0.3340 | +0.1498 | 0.0424 | 0.6568 | 184 |
| prefrontal_rel_gamma | +0.1425 | 0.0537 | 0.3340 | +0.1363 | 0.0650 | 0.6568 | 184 |
| prefrontal_peak_alpha_freq | -0.1413 | 0.0557 | 0.3340 | -0.1097 | 0.1381 | 0.6871 | 184 |
| central_spectral_entropy | +0.1375 | 0.0628 | 0.3340 | +0.1466 | 0.0471 | 0.6568 | 184 |

### Accuracy_Mid_v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **4** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_spectral_entropy | -0.1660 | 0.0243* | 0.7209 | -0.1282 | 0.0830 | 0.6904 | 184 |
| central_rel_delta | +0.1650 | 0.0252* | 0.7209 | +0.1549 | 0.0358 | 0.6904 | 184 |
| central_theta_beta_ratio | +0.1560 | 0.0345* | 0.7209 | +0.1227 | 0.0971 | 0.6904 | 184 |
| right_temporal_rel_theta | +0.1459 | 0.0482* | 0.7209 | +0.1003 | 0.1755 | 0.6904 | 184 |
| left_temporal_rel_theta | +0.1438 | 0.0514 | 0.7209 | +0.0894 | 0.2273 | 0.7387 | 184 |
| left_temporal_theta_beta_ratio | +0.1377 | 0.0623 | 0.7209 | +0.0171 | 0.8180 | 0.9801 | 184 |
| posterior_rel_delta | +0.1369 | 0.0638 | 0.7209 | +0.1491 | 0.0434 | 0.6904 | 184 |
| right_temporal_rel_alpha | -0.1341 | 0.0696 | 0.7209 | -0.1632 | 0.0269 | 0.6904 | 184 |
| right_temporal_theta_beta_ratio | +0.1271 | 0.0855 | 0.7209 | +0.0471 | 0.5251 | 0.9438 | 184 |
| posterior_rel_alpha | -0.1242 | 0.0931 | 0.7209 | -0.1400 | 0.0580 | 0.6904 | 184 |
| central_rel_alpha | -0.1198 | 0.1053 | 0.7209 | -0.1388 | 0.0603 | 0.6904 | 184 |
| left_temporal_abs_gamma | -0.1147 | 0.1212 | 0.7209 | -0.0967 | 0.1916 | 0.6904 | 184 |
| frontal_theta_beta_ratio | +0.1144 | 0.1222 | 0.7209 | +0.0421 | 0.5704 | 0.9438 | 184 |
| central_abs_beta | -0.1142 | 0.1228 | 0.7209 | -0.1424 | 0.0538 | 0.6904 | 184 |
| frontal_abs_delta | +0.1087 | 0.1420 | 0.7209 | +0.0019 | 0.9801 | 0.9801 | 184 |

### Neutral_a

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **14** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| left_temporal_abs_gamma | +0.2757 | 0.0002*** | 0.0139 | +0.1284 | 0.0824 | 0.5262 | 184 |
| left_temporal_abs_beta | +0.2332 | 0.0014** | 0.0658 | +0.1545 | 0.0363 | 0.3304 | 184 |
| prefrontal_abs_gamma | +0.2128 | 0.0037** | 0.1131 | +0.2346 | 0.0013 | 0.1228 | 184 |
| prefrontal_abs_beta | +0.2055 | 0.0051** | 0.1167 | +0.2017 | 0.0060 | 0.1373 | 184 |
| central_abs_beta | +0.1935 | 0.0085** | 0.1549 | +0.1496 | 0.0427 | 0.3530 | 184 |
| central_abs_gamma | +0.1840 | 0.0124* | 0.1702 | +0.1825 | 0.0132 | 0.1747 | 184 |
| frontal_abs_gamma | +0.1826 | 0.0131* | 0.1702 | +0.2107 | 0.0041 | 0.1240 | 184 |
| left_temporal_rel_gamma | +0.1748 | 0.0176* | 0.2008 | +0.0246 | 0.7405 | 0.8632 | 184 |
| right_temporal_abs_beta | +0.1698 | 0.0212* | 0.2146 | +0.1428 | 0.0532 | 0.3921 | 184 |
| left_temporal_spectral_entropy | +0.1662 | 0.0241* | 0.2195 | +0.1181 | 0.1105 | 0.5280 | 184 |
| frontal_abs_beta | +0.1557 | 0.0348* | 0.2878 | +0.1652 | 0.0251 | 0.2535 | 184 |
| left_temporal_rel_beta | +0.1496 | 0.0427* | 0.3212 | +0.1187 | 0.1085 | 0.5280 | 184 |
| posterior_abs_beta | +0.1473 | 0.0460* | 0.3212 | +0.2114 | 0.0040 | 0.1240 | 184 |
| prefrontal_rel_gamma | +0.1451 | 0.0494* | 0.3212 | +0.1245 | 0.0921 | 0.5262 | 184 |
| left_temporal_rel_delta | -0.1290 | 0.0810 | 0.4623 | -0.1411 | 0.0560 | 0.3921 | 184 |

### Neutral_v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **7** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_abs_beta | -0.1987 | 0.0069** | 0.3639 | -0.1991 | 0.0067 | 0.3451 | 184 |
| central_theta_beta_ratio | +0.1884 | 0.0104* | 0.3639 | +0.1278 | 0.0839 | 0.8316 | 184 |
| left_temporal_abs_gamma | -0.1849 | 0.0120* | 0.3639 | -0.1603 | 0.0297 | 0.7244 | 184 |
| left_temporal_abs_beta | -0.1666 | 0.0238* | 0.5413 | -0.1250 | 0.0909 | 0.8316 | 184 |
| prefrontal_theta_beta_ratio | +0.1566 | 0.0337* | 0.5706 | +0.0831 | 0.2623 | 0.8316 | 184 |
| left_temporal_theta_beta_ratio | +0.1532 | 0.0378* | 0.5706 | +0.0363 | 0.6243 | 0.8316 | 184 |
| central_abs_gamma | -0.1487 | 0.0439* | 0.5706 | -0.1963 | 0.0076 | 0.3451 | 184 |
| frontal_abs_beta | -0.1399 | 0.0583 | 0.5917 | -0.1505 | 0.0415 | 0.7545 | 184 |
| frontal_theta_beta_ratio | +0.1378 | 0.0621 | 0.5917 | +0.1238 | 0.0941 | 0.8316 | 184 |
| frontal_abs_gamma | -0.1328 | 0.0723 | 0.5917 | -0.1138 | 0.1242 | 0.8316 | 184 |
| frontal_rel_beta | -0.1302 | 0.0781 | 0.5917 | -0.1189 | 0.1080 | 0.8316 | 184 |
| left_temporal_rel_gamma | -0.1296 | 0.0795 | 0.5917 | -0.0565 | 0.4466 | 0.8316 | 184 |
| central_rel_beta | -0.1255 | 0.0895 | 0.5917 | -0.1201 | 0.1044 | 0.8316 | 184 |
| left_temporal_rel_theta | +0.1241 | 0.0932 | 0.5917 | +0.1046 | 0.1577 | 0.8316 | 184 |
| frontal_rel_gamma | -0.1225 | 0.0975 | 0.5917 | -0.0688 | 0.3531 | 0.8316 | 184 |

### Speed_Max_a

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **6** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| left_temporal_rel_gamma | +0.2322 | 0.0015** | 0.1376 | +0.0996 | 0.1787 | 0.6435 | 184 |
| left_temporal_abs_gamma | +0.2165 | 0.0032** | 0.1436 | +0.2443 | 0.0008 | 0.0757 | 184 |
| left_temporal_abs_beta | +0.1596 | 0.0305* | 0.6014 | +0.1523 | 0.0390 | 0.4440 | 184 |
| frontal_abs_gamma | +0.1591 | 0.0310* | 0.6014 | +0.2004 | 0.0064 | 0.1936 | 184 |
| right_temporal_rel_gamma | +0.1502 | 0.0419* | 0.6014 | +0.0680 | 0.3592 | 0.6856 | 184 |
| left_temporal_spectral_entropy | +0.1485 | 0.0443* | 0.6014 | +0.0760 | 0.3049 | 0.6780 | 184 |
| central_abs_gamma | +0.1432 | 0.0524 | 0.6014 | +0.2174 | 0.0030 | 0.1377 | 184 |
| right_temporal_peak_alpha_freq | -0.1430 | 0.0529 | 0.6014 | -0.1182 | 0.1101 | 0.5896 | 184 |
| left_temporal_rel_beta | +0.1253 | 0.0901 | 0.7546 | +0.0648 | 0.3823 | 0.7072 | 184 |
| prefrontal_abs_delta | +0.1236 | 0.0945 | 0.7546 | +0.0397 | 0.5923 | 0.8693 | 184 |
| frontal_abs_beta | +0.1209 | 0.1021 | 0.7546 | +0.1448 | 0.0498 | 0.5034 | 184 |
| right_temporal_abs_gamma | +0.1196 | 0.1059 | 0.7546 | +0.1916 | 0.0092 | 0.2089 | 184 |
| right_temporal_abs_beta | +0.1189 | 0.1078 | 0.7546 | +0.1318 | 0.0744 | 0.5361 | 184 |
| frontal_abs_theta | +0.1154 | 0.1189 | 0.7726 | +0.0947 | 0.2011 | 0.6435 | 184 |
| right_temporal_rel_beta | +0.1062 | 0.1512 | 0.8573 | +0.0150 | 0.8394 | 0.9643 | 184 |

### Speed_Max_v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **5** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_theta_beta_ratio | +0.1786 | 0.0153* | 0.7627 | +0.1082 | 0.1438 | 0.7991 | 184 |
| left_temporal_rel_gamma | -0.1664 | 0.0240* | 0.7627 | -0.1734 | 0.0186 | 0.4229 | 184 |
| frontal_abs_gamma | -0.1514 | 0.0402* | 0.7627 | -0.0707 | 0.3401 | 0.7991 | 184 |
| left_temporal_theta_beta_ratio | +0.1512 | 0.0405* | 0.7627 | +0.0520 | 0.4830 | 0.8140 | 184 |
| left_temporal_abs_gamma | -0.1482 | 0.0446* | 0.7627 | -0.2206 | 0.0026 | 0.2159 | 184 |
| left_temporal_abs_delta | -0.1363 | 0.0651 | 0.7627 | -0.1379 | 0.0618 | 0.7991 | 184 |
| central_rel_gamma | -0.1361 | 0.0655 | 0.7627 | -0.2073 | 0.0047 | 0.2159 | 184 |
| right_temporal_theta_beta_ratio | +0.1339 | 0.0700 | 0.7627 | +0.0351 | 0.6363 | 0.8711 | 184 |
| central_abs_gamma | -0.1235 | 0.0950 | 0.7627 | -0.1870 | 0.0110 | 0.3350 | 184 |
| frontal_rel_gamma | -0.1229 | 0.0965 | 0.7627 | -0.0979 | 0.1862 | 0.7991 | 184 |
| left_temporal_rel_theta | +0.1219 | 0.0993 | 0.7627 | +0.0700 | 0.3450 | 0.7991 | 184 |
| prefrontal_rel_theta | +0.1213 | 0.1009 | 0.7627 | +0.0931 | 0.2087 | 0.7991 | 184 |
| central_rel_beta | -0.1186 | 0.1090 | 0.7627 | -0.1266 | 0.0868 | 0.7991 | 184 |
| central_spectral_entropy | -0.1128 | 0.1273 | 0.7647 | -0.1192 | 0.1071 | 0.7991 | 184 |
| left_temporal_abs_beta | -0.1115 | 0.1319 | 0.7647 | -0.1142 | 0.1228 | 0.7991 | 184 |

### Speed_Mid_a

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **9** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_abs_gamma | +0.2471 | 0.0007*** | 0.0658 | +0.1725 | 0.0192 | 0.2696 | 184 |
| right_temporal_abs_beta | +0.2243 | 0.0022** | 0.1003 | +0.2077 | 0.0047 | 0.1414 | 184 |
| right_temporal_abs_gamma | +0.2066 | 0.0049** | 0.1486 | +0.2161 | 0.0032 | 0.1414 | 184 |
| frontal_abs_gamma | +0.1701 | 0.0210* | 0.3671 | +0.1585 | 0.0316 | 0.3597 | 184 |
| left_temporal_abs_beta | +0.1666 | 0.0238* | 0.3671 | +0.1334 | 0.0710 | 0.4167 | 184 |
| posterior_abs_beta | +0.1637 | 0.0264* | 0.3671 | +0.1995 | 0.0066 | 0.1508 | 184 |
| central_abs_beta | +0.1618 | 0.0282* | 0.3671 | +0.1516 | 0.0399 | 0.3860 | 184 |
| left_temporal_abs_gamma | +0.1579 | 0.0323* | 0.3674 | +0.1484 | 0.0444 | 0.3860 | 184 |
| prefrontal_abs_delta | +0.1503 | 0.0417* | 0.4218 | +0.0278 | 0.7076 | 0.9558 | 184 |
| posterior_abs_gamma | +0.1380 | 0.0618 | 0.5627 | +0.2292 | 0.0017 | 0.1414 | 184 |
| central_rel_gamma | +0.1243 | 0.0927 | 0.6827 | +0.0105 | 0.8871 | 0.9601 | 184 |
| occipital_abs_theta | +0.1180 | 0.1108 | 0.6827 | +0.0813 | 0.2724 | 0.6886 | 184 |
| central_abs_delta | +0.1142 | 0.1227 | 0.6827 | +0.1254 | 0.0898 | 0.4167 | 184 |
| occipital_abs_beta | +0.1120 | 0.1302 | 0.6827 | +0.1859 | 0.0115 | 0.2098 | 184 |
| posterior_abs_theta | +0.1086 | 0.1422 | 0.6827 | +0.1393 | 0.0594 | 0.4167 | 184 |

### Speed_Mid_v

FDR-significant: **4** / 91 features
Nominally significant (p < .05): **23** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_theta_beta_ratio | +0.2463 | 0.0007*** | 0.0473 | +0.2257 | 0.0021 | 0.0954 | 184 |
| right_temporal_spectral_entropy | -0.2321 | 0.0015** | 0.0473 | -0.1845 | 0.0121 | 0.0954 | 184 |
| right_temporal_theta_beta_ratio | +0.2257 | 0.0021** | 0.0473 | +0.1309 | 0.0766 | 0.2661 | 184 |
| right_temporal_rel_beta | -0.2256 | 0.0021** | 0.0473 | -0.1455 | 0.0488 | 0.2465 | 184 |
| left_temporal_theta_beta_ratio | +0.2102 | 0.0042** | 0.0749 | +0.1317 | 0.0748 | 0.2661 | 184 |
| right_temporal_rel_gamma | -0.2064 | 0.0049** | 0.0749 | -0.1382 | 0.0614 | 0.2612 | 184 |
| left_temporal_rel_gamma | -0.2025 | 0.0058** | 0.0759 | -0.0978 | 0.1866 | 0.3948 | 184 |
| central_rel_beta | -0.1960 | 0.0077** | 0.0873 | -0.2207 | 0.0026 | 0.0954 | 184 |
| posterior_rel_beta | -0.1850 | 0.0119* | 0.1207 | -0.1866 | 0.0112 | 0.0954 | 184 |
| central_rel_gamma | -0.1805 | 0.0142* | 0.1293 | -0.1836 | 0.0126 | 0.0954 | 184 |
| central_spectral_entropy | -0.1716 | 0.0199* | 0.1644 | -0.1996 | 0.0066 | 0.0954 | 184 |
| prefrontal_theta_beta_ratio | +0.1687 | 0.0221* | 0.1673 | +0.1298 | 0.0790 | 0.2661 | 184 |
| occipital_spectral_entropy | -0.1639 | 0.0262* | 0.1730 | -0.1929 | 0.0087 | 0.0954 | 184 |
| frontal_rel_beta | -0.1635 | 0.0266* | 0.1730 | -0.1862 | 0.0114 | 0.0954 | 184 |
| left_temporal_rel_beta | -0.1575 | 0.0327* | 0.1875 | -0.1281 | 0.0830 | 0.2678 | 184 |

## Method 3: Grand Mean

### a

FDR-significant: **1** / 91 features
Nominally significant (p < .05): **10** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| left_temporal_abs_gamma | +0.2687 | 0.0002*** | 0.0206 | +0.1990 | 0.0068 | 0.1026 | 184 |
| central_abs_gamma | +0.2298 | 0.0017** | 0.0772 | +0.2360 | 0.0013 | 0.0714 | 184 |
| left_temporal_abs_beta | +0.2171 | 0.0031** | 0.0932 | +0.1400 | 0.0581 | 0.3582 | 184 |
| frontal_abs_gamma | +0.2067 | 0.0049** | 0.1053 | +0.2071 | 0.0048 | 0.0871 | 184 |
| left_temporal_rel_gamma | +0.2027 | 0.0058** | 0.1053 | +0.0891 | 0.2293 | 0.6152 | 184 |
| right_temporal_abs_beta | +0.1940 | 0.0083** | 0.1263 | +0.1689 | 0.0219 | 0.2492 | 184 |
| right_temporal_abs_gamma | +0.1784 | 0.0154* | 0.2003 | +0.2242 | 0.0022 | 0.0714 | 184 |
| central_abs_beta | +0.1724 | 0.0193* | 0.2191 | +0.1476 | 0.0456 | 0.3582 | 184 |
| prefrontal_abs_gamma | +0.1597 | 0.0304* | 0.3071 | +0.2229 | 0.0024 | 0.0714 | 184 |
| frontal_abs_beta | +0.1500 | 0.0422* | 0.3837 | +0.1408 | 0.0566 | 0.3582 | 184 |
| posterior_abs_beta | +0.1436 | 0.0518 | 0.3950 | +0.1870 | 0.0110 | 0.1431 | 184 |
| left_temporal_spectral_entropy | +0.1427 | 0.0534 | 0.3950 | +0.0808 | 0.2755 | 0.6152 | 184 |
| prefrontal_abs_beta | +0.1406 | 0.0570 | 0.3950 | +0.1444 | 0.0505 | 0.3582 | 184 |
| right_temporal_rel_gamma | +0.1385 | 0.0608 | 0.3950 | +0.1218 | 0.0995 | 0.5657 | 184 |
| occipital_abs_theta | +0.1236 | 0.0947 | 0.5628 | +0.0794 | 0.2839 | 0.6152 | 184 |

### v

FDR-significant: **0** / 91 features
Nominally significant (p < .05): **14** / 91 features

| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |
|-------------|-----------|---|-------|--------------|---|-------|---|
| central_theta_beta_ratio | +0.2516 | 0.0006*** | 0.0520 | +0.1977 | 0.0071 | 0.2388 | 184 |
| left_temporal_theta_beta_ratio | +0.2074 | 0.0047** | 0.1637 | +0.0851 | 0.2507 | 0.7808 | 184 |
| right_temporal_theta_beta_ratio | +0.2043 | 0.0054** | 0.1637 | +0.0944 | 0.2027 | 0.7808 | 184 |
| left_temporal_rel_gamma | -0.1909 | 0.0094** | 0.2149 | -0.1055 | 0.1541 | 0.7383 | 184 |
| left_temporal_abs_gamma | -0.1820 | 0.0134* | 0.2205 | -0.1852 | 0.0119 | 0.2388 | 184 |
| left_temporal_rel_theta | +0.1758 | 0.0170* | 0.2205 | +0.1322 | 0.0737 | 0.6348 | 184 |
| frontal_theta_beta_ratio | +0.1719 | 0.0196* | 0.2205 | +0.1336 | 0.0706 | 0.6348 | 184 |
| central_abs_gamma | -0.1707 | 0.0205* | 0.2205 | -0.2036 | 0.0056 | 0.2388 | 184 |
| central_rel_gamma | -0.1677 | 0.0228* | 0.2205 | -0.1826 | 0.0131 | 0.2388 | 184 |
| central_rel_beta | -0.1652 | 0.0250* | 0.2205 | -0.1834 | 0.0127 | 0.2388 | 184 |
| central_spectral_entropy | -0.1623 | 0.0277* | 0.2205 | -0.1732 | 0.0187 | 0.2835 | 184 |
| prefrontal_theta_beta_ratio | +0.1609 | 0.0291* | 0.2205 | +0.0805 | 0.2775 | 0.7808 | 184 |
| right_temporal_rel_theta | +0.1579 | 0.0323* | 0.2262 | +0.1058 | 0.1529 | 0.7383 | 184 |
| central_abs_beta | -0.1543 | 0.0365* | 0.2374 | -0.1604 | 0.0297 | 0.3856 | 184 |
| prefrontal_rel_theta | +0.1411 | 0.0560 | 0.3399 | +0.0815 | 0.2714 | 0.7808 | 184 |

## Summary: FDR-Significant Results

**9** correlations survived FDR correction:

| Method | DDM Var | EEG Feature | r | p | FDR p |
|--------|---------|-------------|---|---|-------|
| Load/NoLoad | NoLoad_a | left_temporal_abs_gamma | +0.2882 | 0.0001 | 0.0066 |
| PairedMean | Accuracy_Max_a | left_temporal_abs_gamma | +0.2848 | 0.0001 | 0.0081 |
| PairedMean | Neutral_a | left_temporal_abs_gamma | +0.2757 | 0.0002 | 0.0139 |
| Load/NoLoad | NoLoad_v | central_theta_beta_ratio | +0.2705 | 0.0002 | 0.0186 |
| GrandMean | a | left_temporal_abs_gamma | +0.2687 | 0.0002 | 0.0206 |
| PairedMean | Speed_Mid_v | central_theta_beta_ratio | +0.2463 | 0.0007 | 0.0473 |
| PairedMean | Speed_Mid_v | right_temporal_rel_beta | -0.2256 | 0.0021 | 0.0473 |
| PairedMean | Speed_Mid_v | right_temporal_theta_beta_ratio | +0.2257 | 0.0021 | 0.0473 |
| PairedMean | Speed_Mid_v | right_temporal_spectral_entropy | -0.2321 | 0.0015 | 0.0473 |
