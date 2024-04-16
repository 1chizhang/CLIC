import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

bpp_vvc=[




    0.6163272777777775,
    0.5475777777777777,
    0.48784072222222213,
    0.4333699999999998,
    0.38413799999999987,
    0.34118188888888895,
    0.30202522222222217,
    0.2666448333333333,
    0.2357838888888889,
    0.2074918888888888,
    0.18161872222222222,
    0.15942338888888893,
    0.13931327777777777,
    0.1212222222222222,
    0.10556750000000001,




      ]
psnr_vvc = [




    38.171101409034414,
    37.65260644653279,
    37.15087006806389,
    36.63929438189972,
    36.12366537636386,
    35.61691579528249,
    35.099900257328386,
    34.57993698742256,
    34.06776668015085,
    33.54406887605023,
    32.93511769439266,
    32.4185445383782,
    31.894567653525023,
    31.365161795432705,
    30.8466658868599,




     ]



bpp_GLLMM =[0.0730,0.1230,0.2030,0.2971,0.3739,0.4260,0.5225,0.6012]
psnr_GLLMM = [29.95,31.56,33.59,35.38,36.30,36.90,37.81,38.45]

bpp_Entroformer =[0.0817,0.1224, 0.1977,0.2876,0.4067,0.6461,
]

psnr_Entroformer =[29.72,31.19,33.24,34.76,36.35,38.47,
        ]


bpp_Mixed=[0.128, 0.158, 0.221, 0.318, 0.428, 0.605]
psnr_Mixed = [ 32.06, 32.81, 34.35, 35.79, 37.23, 38.7]

bpp_STF = [0.10450155555555554,0.15050133333333338, 0.22135288888888893,0.3195591111111111,0.44326933333333324, 0.6220624444444448


]
psnr_STF = [30.945709566404503,32.38873398167328, 33.85293336207131,35.392414030905705,36.84461655146725,38.366150687822966


]

# bpp_WACNN = [0.1074766666666667, 0.1560015555555556,  0.21731133333333338,0.46767400000000003
# ]
#
#
# PSNR_WACNN=[ 31.11690155480786,32.58426696509282,33.970575604978066, 36.95609633938816
#
# ]

bpp_QResVAE = [0.13276261111111104,
            0.20710416666666664,
            0.2954394444444446,
            0.439147,
            0.6470617777777778,


]
psnr_QResVAE = [  31.491606025312173,
            33.50858662363779,
            34.84809670849707,
            36.789031335903424,
            38.54860285624861,


]

# bpp_cheng2020_attn = [0.098142,0.144817,0.208032,0.288388,0.419039,0.611515]
# psnr_cheng2020_attn = [30.773710,32.433243,33.819927,35.198128,36.757183,38.364407]

bpp_cheng2020_anchor = [0.100168,0.1440802222222222, 0.20095266666666667,0.2924453333333333,0.4072086666666667,0.5537195555555555]
psnr_cheng2020_anchor = [30.413639640808107,31.791408596038817,33.04250720977783,34.7690526008606,36.17197689056397,37.44903198242187]

bpp_Contextformer=[0.096185,
0.146191,
0.198378,
0.283088,
0.38709,
0.436967,
0.620389,]
psnr_Contextformer = [30.94091,
    32.52921,
    33.75947,
    35.25145,
    36.53579,
    37.07024,
    38.44737,]

#



bpp_Ours = [ 0.11085955555555557,0.1530044444444445,0.21784422222222222,0.3075024444444444,0.4302546666666667,0.5986468888888888
]
psnr_Ours = [31.25474229011299,32.80054725616113,34.305446123065806,35.64734371365596,37.09071200827428, 38.49907830633111
             ]



plt.rcParams['pdf.fonttype'] = 42
markersize = 6
linewidth = 2
fig, ax = plt.subplots(figsize=(6.5*2,6.5*2))
plt.plot(bpp_vvc, psnr_vvc, label='VVC',
         color='yellow', linewidth=linewidth, )
plt.plot(bpp_cheng2020_anchor, psnr_cheng2020_anchor, label='Cheng2020 (CVPR2020)',
         color='blue', linewidth=linewidth,)
#
plt.plot(bpp_Entroformer, psnr_Entroformer, label='Entroformer (ICLR2022)',
         color='#808000', linewidth=linewidth, )


#
plt.plot(bpp_STF, psnr_STF, label='STF (CVPR2022)',
         color='r', linewidth=linewidth,)
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
# plt.plot(bpp_Ours_w_o_g, psnr_Ours_w_o_g, label='PointCom_W/O Guided',
#          color='black', linewidth=linewidth, marker='*', markersize=markersize)
ax.locator_params(axis='x', nbins=10)
ax.locator_params(axis='y', nbins=10)
plt.tick_params(labelsize=36)
plt.xlabel("bpp", fontdict={ 'weight':'normal','size':40})
plt.ylabel("PSNR (dB)", fontdict={ 'weight':'normal','size':40})
plt.grid(ls='-.')
plt.legend(loc='lower right', ncol=1,
           prop={ 'weight':'normal','size':26})
fig.tight_layout()
os.makedirs('./fig/', exist_ok=True)
figname = os.path.join('./fig', 'RD_tecnick_mixed.pdf')
fig.savefig(figname)
print(figname)