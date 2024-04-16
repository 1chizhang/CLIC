import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

bpp_vvc=[0.113366021
,0.133433024
,0.15636105
,0.183322483
,0.213667128
,0.247949388
,0.288040161
,0.331813388
,0.380132039
,0.435056051
,0.494663662
,0.559277852
,0.631427341



      ]
psnr_vvc = [
11.07839182
,11.6061411
,12.128885
,12.69969869
,13.27113371
,13.83963594
,14.45294274
,15.05100987
,15.6421448
,16.25824998
,16.85120208
,17.44076371
,18.02505357


     ]



bpp_GLLMM =[0.0741,0.1871,0.3587,0.5968]
psnr_GLLMM = [11.95,15.38,18.63,21.12]

bpp_cheng2020 = [0.09807332356770833,0.14954969618055552,0.2262946234809028,0.31735907660590273,0.44893052842881936, 0.5892096625434028

      ]

psnr_cheng2020 = [12.491271480983675,14.123669420068836, 15.937395218749687,17.552708478108574,19.199070794178283,20.825236298221117



     ]

# bpp_Mixed=[]
# psnr_Mixed = [ ]

bpp_STF = [ 0.109,
    0.168,
    0.236,
    0.340,
    0.483,
    0.664]
psnr_STF = [ 13.63,
    14.96,
    16.49,
    18.17,
    20.19,
    21.50]

bpp_WACNN = [ 0.115,
    0.177,
    0.254,
    0.361,
    0.496,
    0.684]


PSNR_WACNN=[13.79,
    15.12,
    16.93,
    18.48,
    20.16,
    21.69

]






# bpp_NeuralSyntax = [0.3855,
# 0.6617 ,
#
#  ]
# PSNR_NeuralSyntax = [  18.3834,21.3182]


bpp_Entroformer = [0.1017,0.1572, 0.1975, 0.2288, 0.3286, 0.4701, 0.6686, ]
psnr_Entroformer = [12.56020135
,14.2794201
,15.29589509
,16.02059991
,17.71086594
,19.48847478
,21.33712661
]

bpp_ELIC = [0.08924,
0.21489,
0.29947,
0.39211,
0.66574

]
psnr_ELIC = [12.2195,
15.836,
17.2579,
18.8087,
21.6487
]


bpp_Contextformer=[0.10925,
0.16861,
0.22073,
0.31243,
0.42828,
0.48729,
0.69833
]
psnr_Contextformer = [12.9623,
14.7872,
16.0714,
17.85,
19.5029,
20.2026,
22.1264
]

#



bpp_Ours = [ 0.11770969,	0.167907715,	0.237921821,	0.333984375,	0.463243273,	0.642890082
 ]
psnr_Ours = [ 13.73338775,	15.24368153,	16.85315645,	18.53260855,	20.21016797,	21.92905226
]




markersize = 6
linewidth = 2
fig, ax = plt.subplots(figsize=(6.5*2,8*2))
plt.plot(bpp_vvc, psnr_vvc, label='VVC',
         color='yellow', linewidth=linewidth, )
plt.plot(bpp_cheng2020, psnr_cheng2020, label='Cheng2020(CVPR2020)',
         color='blue', linewidth=linewidth,)

plt.plot(bpp_Entroformer, psnr_Entroformer, label='Entroformer(ICLR2022)',
         color='#808000', linewidth=linewidth, )
plt.plot(bpp_ELIC, psnr_ELIC, label='ELIC(CVPR2022)',
         color='#008080', linewidth=linewidth, )

plt.plot(bpp_STF, psnr_STF, label='STF(CVPR2022)',
         color='r', linewidth=linewidth,)
plt.plot(bpp_WACNN, PSNR_WACNN, label='WACNN(CVPR2022)',
         color='pink', linewidth=linewidth)
plt.plot(bpp_Contextformer, psnr_Contextformer, label='Contextformer(ECCV2022)',
         color='#00FFFF', linewidth=linewidth)


plt.plot(bpp_GLLMM, psnr_GLLMM, label='GLLMM(TIP2023)',
         color='gray', linewidth=linewidth,)
# plt.plot(bpp_Mixed, psnr_Mixed, label='Mixed(CVPR2023)',
#          color='orange', linewidth=linewidth,)
plt.plot(bpp_Ours, psnr_Ours, label='Ours',
         color='black', linewidth=linewidth, marker='o', markersize=markersize)
# plt.plot(bpp_Ours_QVRF, psnr_Ours_QVRF, label='PointCom_QVRF',
#          color='black', linewidth=linewidth, marker='*', markersize=markersize)
ax.locator_params(axis='x', nbins=10)
ax.locator_params(axis='y', nbins=10)
plt.tick_params(labelsize=36)
plt.xlabel("bpp", fontdict={ 'weight':'normal','size':40})
plt.ylabel("MS-SSIM (dB)", fontdict={ 'weight':'normal','size':40})
plt.grid(ls='-.')
plt.legend(loc='lower right', ncol=1,
           prop={ 'weight':'normal','size':20})
fig.tight_layout()
os.makedirs('./fig/', exist_ok=True)
figname = os.path.join('./fig', 'RD_SSIM_sup.pdf')
fig.savefig(figname)
print(figname)