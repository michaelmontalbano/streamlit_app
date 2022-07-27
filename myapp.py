import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex
from settings import MESH_colors, MESH_bounds
import metrics


st.title('Model Prediction Metrics')
#model = st.radio("Pick a model.")
number = st.number_input("Pick a sample number (0-939)", 0, 939)
cutoff = st.number_input("Pick a cutoff value")
multiplier = st.number_input('Pick a multiplier for beta')
k_pct = st.number_input('Pick a percentage for k')
beta = 12960000*multiplier

y_true = load('data/y_true.npy').squeeze()
y_pred = load('data/y_pred_1.npy').squeeze()

f, axs = plt.subplots(1,2,figsize=(16,8))

img_true = y_true[number]
img_pred = y_pred[number]

plt.subplot(121)
ax = plt.gca()
cs = plt.contourf(img_true,levels=MESH_bounds,colors=MESH_colors, extend='both',orientation='horizontal', shrink=0.5, spacing='proportional')   
plt.colorbar(cs, ticks=MESH_bounds)
f.tight_layout(pad=3.0)
plt.ylabel('y (1/2 km)')
plt.xlabel('x (1/2 km)')
plt.xlim([0,60])
plt.xticks([0,10,20,30,40,50,60])
plt.yticks([0,10,20,30,40,50,60])
plt.title(f'True MESH  #{number} (mm)')

plt.subplot(122)
ax = plt.gca()
cs = plt.contourf(img_pred,levels=MESH_bounds,colors=MESH_colors, extend='both',orientation='horizontal', shrink=0.5, spacing='proportional')   
plt.colorbar(cs, ticks=MESH_bounds)
f.tight_layout(pad=3.0)
plt.ylabel('y (1/2 km)')
plt.xlabel('x (1/2 km)')
plt.xlim([0,60])
plt.xticks([0,10,20,30,40,50,60])
plt.yticks([0,10,20,30,40,50,60])
plt.title(f'Predicted MESH with MSE #{number} (mm)')
st.pyplot(f)

far_funct = metrics.FAR(cutoff)
pod_funct = metrics.POD(cutoff)

# binarize A and B
A = np.where(img_true > cutoff, 1, 0)
B = np.where(img_pred > cutoff, 1, 0)
min_nonzero = min(A.sum(), B.sum())
# get mean squared error of img_true and img_pred
mse = np.mean((img_true - img_pred)**2)
if min_nonzero > 1:
    far = far_funct(A,B)
    pod = pod_funct(A,B)
    mindists_AB = np.asarray(metrics.min_dists(img_true, img_pred, cutoff))
    mindists_BA = np.asarray(metrics.min_dists(img_pred, img_true, cutoff))
    hausdorf_distance = metrics.hausdorf(mindists_AB, mindists_BA)
    phdk_distance = metrics.PHDK(mindists_AB, mindists_BA, k_pct)
    gbeta = metrics.G_beta(A, B, mindists_AB, mindists_BA, beta)
else:
    far = 0
    pod = 0
    hausdorf_distance = -99
    phdk_distance = -99
    gbeta = -99
metrics_dict = {'MSE': mse, 'FAR': [far], 'POD': [pod], 'Hausdorff': [hausdorf_distance], 'PHDK': [phdk_distance], 'Gbeta': [gbeta]}
df = pd.DataFrame(metrics_dict)

st.table(df)
