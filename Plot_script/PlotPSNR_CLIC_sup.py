import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
# Kodak

bpp_vvc=[


      # 0.9423138766004816,
      # 0.8444196360771851,
      0.7538423408402919,
      0.6702595201051206,
      0.59518510157408,
      0.5252932320898265,
      0.46076541450667124,
      0.40504563627128026,
      0.35592636588016,
      0.31253131014472113,
      0.2748262992564997,
      0.24058864247766917,
      0.20950151171558157,
      0.18268486463767655,
      0.15839832018867633,
      0.13655582141082556,
      0.117793075335131,


      ]
psnr_vvc = [

    # 38.89546924857088,
    # 38.32573137930149,
    37.736443604899804,
    37.14331315204327,
    36.565881698336945,
    35.97617273121685,
    35.38061499670864,
    34.81202311878888,
    34.249322119734664,
    33.6939776202705,
    33.163073187873266,
    32.62600126226385,
    32.06089084045758,
    31.540181805015788,
    31.01228174948926,
    30.483128273917607,
    29.972941975821747,



     ]



bpp_cheng2020= [ 0.10928561665620927,0.15923629619593963,0.22194610251593735, 0.33531310846037665, 0.4752075747098571,0.653677332643557
      ]

psnr_cheng2020= [29.488278325398763,30.79480152130127,31.945448366800942, 33.75064366658528,35.11817162831624, 36.38879547119141



     ]

bpp_STF = [0.11126620739356544,0.16234939199624682,0.24662961811523365,0.3633025883536651,0.5171482620270895, 0.7477276625277806

]
psnr_STF = [ 29.943431922825845,31.279213052169027, 32.73747691476855,34.359618238837896,35.96272711833103,37.68663146626977

]



bpp_QResVAE = [0.15402733115215544,
            0.24311623438743718,
            0.354507456532663,
            0.5405831677901929,
            0.7976627939366602,
            # 1.101979867273523,

]
psnr_QResVAE = [ 30.671247891019465,
            32.713007183985695,
            34.13366588102288,
            36.28934198902916,
            38.244548978666906,
            # 40.244713416999524,

]

bpp_Entroformer = [0.0852,0.1330,0.2264,0.3351,0.4871,0.7887]
psnr_Entroformer = [28.72,30.17,32.21,33.76,35.46,37.76]





#



bpp_Ours = [ 0.12222157605156542,0.17247691468041265,0.24709585408272056,0.35340590641861913,0.5168577634885381, 0.7279586008258536
             ]
psnr_Ours = [30.331743763209126,31.81836284807438, 33.28672754622368,34.67256597889538,36.29855731002758, 37.903782790207444
 ]
#



markersize = 6
linewidth = 2
fig, ax = plt.subplots(figsize=(6.5*2,8*2))
plt.plot(bpp_vvc, psnr_vvc, label='VVC',
         color='yellow', linewidth=linewidth, )
plt.plot(bpp_cheng2020, psnr_cheng2020, label='Cheng2020 (CVPR2020)',
         color='blue', linewidth=linewidth,)

plt.plot(bpp_Entroformer, psnr_Entroformer, label='Entroformer (ICLR2022)',
         color='#808000', linewidth=linewidth, )



plt.plot(bpp_STF, psnr_STF, label='STF (CVPR2022)',
         color='r', linewidth=linewidth,)


plt.plot(bpp_QResVAE, psnr_QResVAE, label='QResVAE (WACV2023Best)',
         color='purple', linewidth=linewidth,)
#
plt.plot(bpp_Ours, psnr_Ours, label='Ours',
         color='black', linewidth=linewidth, marker='o', markersize=markersize)

ax.locator_params(axis='x', nbins=10)
ax.locator_params(axis='y', nbins=10)
plt.tick_params(labelsize=36)
plt.xlabel("bpp", fontdict={ 'weight':'normal','size':40})
plt.ylabel("PSNR (dB)", fontdict={ 'weight':'normal','size':40})
plt.grid(ls='-.')
plt.legend(loc='lower right', ncol=1,
           prop={ 'weight':'normal','size':20})
fig.tight_layout()
os.makedirs('./fig/', exist_ok=True)
figname = os.path.join('./fig', 'RD_CLIC_sup.pdf')
fig.savefig(figname)
print(figname)