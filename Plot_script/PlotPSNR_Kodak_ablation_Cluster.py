import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt






bpp_9centers = [0.203091092
 ]
PSNR_9centers = [30.86312206
 ]

bpp_Eu = [0.202470]
psnr_Eu = [30.8590489]
bpp_w_o_ckpt_cluster = [ 0.203999837
]
psnr_w_o_ckpt_cluster = [ 30.82966858
]

bpp_w_o_mlp_offset = [0.201531245
]
psnr_w_o_mlp_offset = [30.8407
]


bpp_iterative=[0.201113143
]
psnr_iterative = [30.9083
]

#
bpp_overlapped = [0.2017

]
psnr_overlapped = [30.93214374

]

bpp_soft = [0.197743734

]
psnr_soft = [30.72056101

]

bpp_Ours = [  0.12795342339409724, 0.20016140407986113, 0.3034294976128472]
psnr_Ours = [ 29.24774588024061, 30.902700198688795,32.584972281153256]



plt.rcParams['pdf.fonttype'] = 42
markersize = 30
linewidth = 4
fig, ax = plt.subplots(figsize=(7*2,6.5*2))
plt.xlim((0.195,0.210))
plt.ylim((30.5,31.1))


plt.plot(bpp_w_o_ckpt_cluster, psnr_w_o_ckpt_cluster, label='w/o CP CCB',
         color='purple', linewidth=linewidth,marker='x', markersize=markersize)
plt.plot(bpp_w_o_mlp_offset, psnr_w_o_mlp_offset, label='w/o MLP Offset',
         color='gray', linewidth=linewidth,marker='+', markersize=markersize)
plt.plot(bpp_iterative, psnr_iterative, label='MLP → Iterative Offset',
         color='orange', linewidth=linewidth,marker='^', markersize=markersize)
plt.plot(bpp_overlapped, psnr_overlapped, label='Non-overlapped → Overlapped',
         color='#000080', linewidth=linewidth, marker='v', markersize=markersize)
plt.plot(bpp_soft, psnr_soft, label='Points → Similarity Normalize',
         color='#008080', linewidth=linewidth, marker='*', markersize=markersize)

plt.plot(bpp_Eu, psnr_Eu, label='Cosine → Eu Distance',
         color='#2D6FEA', linewidth=linewidth,marker='d', markersize=markersize)

plt.plot(bpp_9centers, PSNR_9centers, label='Four → Nine Centers',
         color='#00FFFF', linewidth=linewidth,marker='8', markersize=markersize)
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
figname = os.path.join('./fig', 'RD_ablation_on_Cluster.pdf')
fig.savefig(figname)
print(figname)