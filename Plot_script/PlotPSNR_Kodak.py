import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

bpp_vvc=[


    0.9774008856879339,
    0.8805211385091146,
    0.7897109985351562,
    0.7076187133789061,
    0.6296301947699652,
    0.5570797390407985,
    0.49359554714626735,
    0.43371582031250017,
    0.3787095811631944,
    0.33100043402777773,
    0.28728654649522567,
    0.2476306491427952,
    0.21332126193576387,
    0.1826756795247396,
    0.15583123101128474,
    0.13300577799479166,
    0.11299387613932294,


      ]
psnr_vvc = [


    38.0919141131403,
    37.47104854011085,
    36.84607932760226,
    36.22182842219978,
    35.585510026803135,
    34.94384492591835,
    34.33035268646733,
    33.695655397276234,
    33.06791344221022,
    32.47117182551503,
    31.870141804054715,
    31.26421523513569,
    30.690378199066853,
    30.12833268890773,
    29.577607191514634,
29.054922843407308,
    28.537732973445078,



     ]



bpp_GLLMM =[0.0920,0.1556,0.2822,0.4408,0.5655,0.6458,0.7921,0.9060]
psnr_GLLMM = [27.99,29.63,31.82,33.97,35.25,35.96,37.06,37.85]

# bpp_cheng2020 = [ 0.119837,
# 0.191777,
# 0.291133,
# 0.415871,
# 0.626334,
# 0.922226
#
#       ]
#
# psnr_cheng2020 = [28.696959,
#     30.316946,
# 31.771183,
# 33.266029,
# 35.291298,
# 37.526440
#
#
#
#      ]

# 4.8353950427513
bpp_cheng2020 = [ 0.1195407443576389,
0.18382093641493055,
0.2709927029079861,
0.417300754123264,
0.5944349500868055,
0.8057996961805554

      ]

psnr_cheng2020 = [28.575154225031536,
29.963903188705444,
31.33073075612386,
33.375502506891884,
35.09538650512695,
36.67982419331869

     ]




bpp_Mixed=[0.155, 0.194, 0.300, 0.443, 0.625, 0.880]
psnr_Mixed = [ 30.07, 30.85, 32.59, 34.33, 36.15, 38.07]

bpp_STF = [0.124,
    0.191,
    0.298,
    0.441,
    0.651,
    0.903]
psnr_STF = [29.14,30.50,32.15,
    33.97,
    35.82,
    37.72]

bpp_WACNN = [0.127,
    0.199,
    0.309,
    0.449,
    0.649,
    0.895]


PSNR_WACNN=[
29.22,
    30.59,
    32.26,
    34.15,
    35.91,
    37.72
]

bpp_QResVAE = [0.18335469563802084,
            0.3011873033311632,
            0.4519933064778645,
            0.6737916734483508,
            0.9541693793402777,
]
psnr_QResVAE = [ 30.02302909544619,
            31.980835998050456,
            33.89135182243866,
            36.1122462862753,
            38.166281526220864,
]





bpp_NeuralSyntax = [0.1868,0.2875 ,0.3950,0.7532,1.0548 ]
PSNR_NeuralSyntax = [ 30.2688,31.8869, 33.2138, 36.5635,38.4772]


bpp_Entroformer = [  0.0866,0.1452,0.2632,0.4058,0.5925,0.9313]
psnr_Entroformer = [ 27.63,29.16,31.38,33.18,35.13,37.72]

bpp_ELIC = [
    0.12416, 0.19611, 0.33294, 0.49021, 0.70347, 0.8572
    # 1.1
]
psnr_ELIC = [
    29.1294
    , 30.7081
    , 32.7992
    , 34.5827, 36.4756, 37.6235
]


bpp_Contextformer=[0.11206
,0.18891,
0.27318,
0.40964,
0.57055,
0.64757,
0.88783]
psnr_Contextformer = [28.8241,
30.5508,
31.9514,
33.7537,
35.4501,
36.0995,
37.7272]

#



bpp_Ours = [  0.12795342339409724, 0.20016140407986113, 0.3034294976128472, 0.44619072808159715,0.6339280870225693,0.8727552625868057]
psnr_Ours = [ 29.24774588024061, 30.902700198688795,32.584972281153256, 34.315261821745736, 36.13837189689688,37.937971608884354]



plt.rcParams['pdf.fonttype'] = 42
markersize = 6
linewidth = 2
fig, ax = plt.subplots(figsize=(6.5*2,6.5*2))
plt.plot(bpp_vvc, psnr_vvc, label='VVC',
         color='yellow', linewidth=linewidth, )
plt.plot(bpp_cheng2020, psnr_cheng2020, label='Cheng2020 (CVPR2020)',
         color='blue', linewidth=linewidth,)

plt.plot(bpp_Entroformer, psnr_Entroformer, label='Entroformer (ICLR2022)',
         color='#808000', linewidth=linewidth, )
plt.plot(bpp_ELIC, psnr_ELIC, label='ELIC (CVPR2022)',
         color='#008080', linewidth=linewidth, )
plt.plot(bpp_NeuralSyntax, PSNR_NeuralSyntax, label='NeuralSyntax (CVPR2022)',
         color='#ff00ff', linewidth=linewidth)

plt.plot(bpp_STF, psnr_STF, label='STF (CVPR2022)',
         color='r', linewidth=linewidth,)
plt.plot(bpp_WACNN, PSNR_WACNN, label='WACNN (CVPR2022)',
         color='pink', linewidth=linewidth)
plt.plot(bpp_Contextformer, psnr_Contextformer, label='Contextformer (ECCV2022)',
         color='#00FFFF', linewidth=linewidth)

plt.plot(bpp_QResVAE, psnr_QResVAE, label='QResVAE (WACV2023Best)',
         color='purple', linewidth=linewidth,)
plt.plot(bpp_GLLMM, psnr_GLLMM, label='GLLMM (TIP2023)',
         color='gray', linewidth=linewidth,)
plt.plot(bpp_Mixed, psnr_Mixed, label='Mixed(CVPR2023)',
         color='orange', linewidth=linewidth,)
plt.plot(bpp_Ours, psnr_Ours, label='Ours',
         color='black', linewidth=linewidth, marker='o', markersize=markersize)

ax.locator_params(axis='x', nbins=10)
ax.locator_params(axis='y', nbins=10)
plt.tick_params(labelsize=36)
plt.xlabel("bpp", fontdict={ 'weight':'normal','size':40})
plt.ylabel("PSNR (dB)", fontdict={ 'weight':'normal','size':40})
plt.grid(ls='-.')
plt.legend(loc='lower right', ncol=1,
           prop={'weight':'normal','size':26})
fig.tight_layout()
os.makedirs('./fig/', exist_ok=True)
figname = os.path.join('./fig', 'RD_mixed.pdf')
fig.savefig(figname)
print(figname)