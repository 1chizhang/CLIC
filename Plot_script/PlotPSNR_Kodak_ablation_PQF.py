import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


#


bpp_Ours_w_dep = [0.205434163411458
]
psnr_Ours_w_dep = [30.90901533

]

bpp_Ours_w_o_swin = [0.205813938
]
psnr_Ours_w_o_swin = [30.97446987

]
psnr_ours_w_o_attn = [30.52308081,]

bpp_ours_w_o_attn = [0.201833089]
bpp_Ours = [  0.12795342339409724, 0.20016140407986113, 0.3034294976128472]
psnr_Ours = [ 29.24774588024061, 30.902700198688795,32.584972281153256]

bpp_Ours_w_PQF = [0.2005]
psnr_Ours_w_PQF = [30.80346]

bpp_Ours_w_o_PQF = [0.20015]
psnr_Ours_w_o_PQF = [30.7372692879204]

bpp_w_o_pe=[0.202836778
]
psne_w_o_pe= [30.84026198
]
bpp_attn_replaced = [0.204104953
]
psnr_attn_relaced = [30.82834174430859]

plt.rcParams['pdf.fonttype'] = 42
markersize = 30

linewidth = 4
# fig, ax = plt.subplots(figsize=(6.5*2,5.85*2))
fig, ax = plt.subplots(figsize=(7*2,6.5*2))

plt.xlim((0.195,0.21))
plt.ylim((30.5,31.1))

# plt.plot(bpp_Ours_w_dep, psnr_Ours_w_dep, label='Clu → Conv',
#          color='#008080', linewidth=linewidth,marker='.', markersize=markersize)

# plt.plot(bpp_Ours_w_o_swin, psnr_Ours_w_o_swin, label='Clu → W-MHSA',
#          color='gray', linewidth=linewidth, marker='>', markersize=markersize)
# plt.plot(bpp_ours_w_o_attn, psnr_ours_w_o_attn, label='w / o Attn',
#          color='red', linewidth=linewidth, marker='<', markersize=markersize)
# plt.plot(bpp_attn_replaced, psnr_attn_relaced, label='Attn → Conv',
#          color='green', linewidth=linewidth, marker='s', markersize=markersize)
plt.plot(bpp_w_o_pe, psne_w_o_pe, label='w/o PE',
         color='purple', linewidth=linewidth, marker='d', markersize=markersize)
plt.plot(bpp_Ours_w_o_PQF, psnr_Ours_w_o_PQF, label='w/o PQF',
         color='orange', linewidth=linewidth,marker='v', markersize=markersize)

plt.plot(bpp_Ours_w_PQF, psnr_Ours_w_PQF, label='GuidedPQF → NormalPQF',
         color='blue', linewidth=linewidth, marker='*', markersize=markersize)

plt.plot(bpp_Ours, psnr_Ours, label='Ours',
         color='black', linewidth=linewidth, marker='o', markersize=markersize)
ax.locator_params(axis='x', nbins=5)
ax.locator_params(axis='y', nbins=5)
plt.tick_params(labelsize=36)
plt.xlabel("bpp", fontdict={ 'weight':'normal','size':40})
plt.ylabel("PSNR (dB)", fontdict={ 'weight':'normal','size':40})
plt.grid(ls='-.')
plt.legend(loc='lower right', ncol=1,
           prop={ 'weight':'normal','size':32})
fig.tight_layout()
os.makedirs('./fig/', exist_ok=True)
figname = os.path.join('./fig', 'RD_ablation_on_PQF.pdf')
fig.savefig(figname)
print(figname)