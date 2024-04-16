import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


#



bpp_Ours = [  0.12795342339409724, 0.20016140407986113, 0.3034294976128472, 0.44619072808159715,0.6339280870225693,0.8727552625868057]
psnr_Ours = [ 29.24774588024061, 30.902700198688795,32.584972281153256, 34.315261821745736, 36.13837189689688,37.937971608884354]



bpp_Ours_w_o_attn = [0.127187093,	0.201833089,	0.308681912,	0.456882053,	0.646396213,	0.886956109
]
psnr_Ours_w_o_attn = [28.88861804,	30.52308081,	32.21674003,	34.07127284,	35.95008885,	37.79774435
]




markersize = 8
linewidth = 2
fig, ax = plt.subplots(figsize=(6.5*2,6*2))


plt.plot(bpp_Ours_w_o_attn, psnr_Ours_w_o_attn, label='w / o Attention Enhanced',
         color='orange', linewidth=linewidth,marker='v', markersize=markersize)

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
figname = os.path.join('./fig', 'RD_ablation_on_Attn.pdf')
fig.savefig(figname)
print(figname)