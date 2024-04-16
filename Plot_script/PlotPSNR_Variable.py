import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


#



bpp_Ours = [  0.12795342339409724, 0.20016140407986113, 0.3034294976128472, 0.44619072808159715,0.6339280870225693,0.8727552625868057]
psnr_Ours = [ 29.24774588024061, 30.902700198688795,32.584972281153256, 34.315261821745736, 36.13837189689688,37.937971608884354]



bpp_Variable = [0.148529053,	0.216054281,	0.315266927,	0.448330349,	0.610788981,	0.813835992

]
psnr_Variable = [29.63892624,	31.1453975,	32.75745261,	34.42642069,	35.98781157,	37.38005075

]


bpp_Ours_tec = [  0.11085955555555557,0.1530044444444445,0.21784422222222222,0.3075024444444444,0.4302546666666667,0.5986468888888888]
psnr_Ours_tec  = [ 31.25474229011299,32.80054725616113,34.305446123065806,35.64734371365596,37.09071200827428, 38.49907830633111]


bpp_Ours_tec_var = [  0.1178557777777778,
      0.16320688888888887,
      0.227784,
      0.31360200000000005,
      0.4181384444444445,
      0.551369111111111
]
psnr_Ours_tec_var  = [
31.43698742
,32.95804118
,34.43852329
,35.80671208
,36.99694666
,38.02269536
]


bpp_Ours_clic = [  0.12222157605156542,0.17247691468041265,0.24709585408272056,0.35340590641861913,0.5168577634885381, 0.7279586008258536]
psnr_Ours_clic  = [ 30.331743763209126,31.81836284807438, 33.28672754622368,34.67256597889538,36.29855731002758, 37.903782790207444]


bpp_Ours_clic_var = [0.13363312739625127,
      0.18781300577844676,
      0.2659947360639781,
      0.3705433821464229,
      0.4990590445471532,
      0.6616271735470426
]
psnr_Ours_clic_var  = [30.52950989
,
      31.98423589
,
      33.47682071
,
      34.8914134
,
      36.18742762
,
      37.30912413
]





markersize = 8
linewidth = 2
fig, ax = plt.subplots(figsize=(6.5*2,6.5*2))
plt.rcParams['pdf.fonttype'] = 42
# plt.xlim((0.18,0.23))
# plt.ylim((0,31.3))




plt.plot(bpp_Ours, psnr_Ours, label='Ours (Kodak)',
         color='black', linewidth=linewidth, marker='o', markersize=markersize)
plt.plot(bpp_Variable, psnr_Variable, label='Ours + QVRF (Kodak)',
         color='#7D3C98', linewidth=linewidth, markersize=markersize)


plt.plot(bpp_Ours_tec, psnr_Ours_tec, label='Ours (Tecnick)',
         color='orange', linewidth=linewidth,marker='p', markersize=markersize)
plt.plot(bpp_Ours_tec_var, psnr_Ours_tec_var, label='Ours + QVRF (Tecnick)',
         color='gray', linewidth=linewidth, markersize=markersize)

plt.plot(bpp_Ours_clic, psnr_Ours_clic, label='Ours (CLIC2022)',
         color='#808000', linewidth=linewidth,marker='P', markersize=markersize )

plt.plot(bpp_Ours_clic_var, psnr_Ours_clic_var, label='Ours + QVRF (CLIC2022)',
         color='r', linewidth=linewidth,markersize=markersize)

ax.locator_params(axis='x', nbins=10)
ax.locator_params(axis='y', nbins=10)
plt.tick_params(labelsize=36)
plt.xlabel("bpp", fontdict={ 'weight':'normal','size':40})
plt.ylabel("PSNR (dB)", fontdict={ 'weight':'normal','size':40})
plt.grid(ls='-.')
plt.legend(loc='lower right', ncol=1,
           prop={ 'weight':'normal','size':30})
fig.tight_layout()
os.makedirs('./fig/', exist_ok=True)
figname = os.path.join('./fig', 'RD_Variable.pdf')
fig.savefig(figname)
print(figname)