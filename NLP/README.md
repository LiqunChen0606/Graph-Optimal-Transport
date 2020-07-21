# Graph_in_NLP

## Summary
In this work, Fused Gromov-Wasserstein distance is applied to improve the performance of different NLP tasks, such as Machine Translation, abstractive summarization, etc

Gromov-Wasserstein distance only or Wasserstein distance only also tested, the machine translation task on EN-VI dataset simply shows FGWD is better than both W or GW.

| Modle        | EN-VI uncased| EN-VI cased  | EN-DE uncased | EN-DE cased|
| ------------- |:-------------:| -----:|-----: | -----: |
| transformer base     | 29.25   | 28.46 | 25.60 | 25.12 |
| transformer + W      | 29.49   | 28.68 | TBD |TBD |
| transformer + GW     | 28.65   | 28.34 | TBD |TBD |
| transformer + FGW    | 29.92   | 29.09 | 26.05 |25.54 |

 
## Brief Introduction
Wasserstein distance: 
![alt text](https://github.com/LiqunChen0606/Graph_in_NLP/blob/master/PIC/W.png)
<!-- ![alt text](https://raw.githubusercontent.com/LiqunChen0606/Graph_in_NLP/master/PIC/W.png?token=ADIFJNUFY4QT53MAWAI2DK25SYJEA) -->

Gromov-Wasserstein distance: 
![alt text](https://github.com/LiqunChen0606/Graph_in_NLP/blob/master/PIC/GW.png)

Fused Gromov-Wasserstein distance: 
![alt text](https://github.com/LiqunChen0606/Graph_in_NLP/blob/master/PIC/FGW.png)
