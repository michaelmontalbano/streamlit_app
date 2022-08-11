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

# 2: loss-mse_dataset-shave_L2-0.01_drop-0.1_junct-Add_filters-64f128_act-lrelu_cut-30_transpose-1_gain-0.0_bias-0.0_init-normal_variant-unetpp_block-vanilla_exp_index-7_kernel-3_out-act-relu_results.pkl

models = ['2','6','7','10']
datasets = ['all','severe','sig-severe']
st.title('Model Prediction Metrics')
model = st.radio("Pick a model.", models)
dataset = st.radio("Pick a dataset.", datasets)
cutoff = st.number_input("Pick a cutoff value")
multiplier = st.number_input('Pick a multiplier for beta')
k_pct = st.number_input('Pick a percentage for k')
beta = 12960000*multiplier
number = st.number_input("Pick a sample number (0-939)", 0, 939)


coeficients = st.text_input('Enter the coefficients')


# convert coefficients to list of comma separated floats
coeficients = coeficients.split(',')
cCSI, cHaus, cPHDK, cGbeta, cDelta, cG, cZhu, cFA, cMiss = [float(i) for i in coeficients]

y_true = load('data/y_true.npy').squeeze()
y_pred = load(f'data/y_pred_{model}.npy').squeeze()

indices = range(len(y_true))
if dataset == 'severe':
    # keep only images where each image in y_true has max above 25
    indices = [x for x,y in zip(range(len(y_true)),y_true) if np.max(y) > 25]
if dataset == 'sig-severe':
    indices = [x for x,y in zip(range(len(y_true)),y_true) if np.max(y) > 50]
y_true = y_true[indices,:,:]
y_pred = y_pred[indices,:,:]

f, axs = plt.subplots(1,2,figsize=(16,8))

img_true = y_true[number]
img_pred = y_pred[number]

# img_true = np.rot90(img_true)
# img_pred = np.rot90(img_pred)

# img_true = np.rot90(img_true)
# img_pred = np.rot90(img_pred)

plt.subplot(121)
cs = plt.contourf(img_true,levels=MESH_bounds,colors=MESH_colors, extend='both', shrink=0.5, origin=None)   
plt.colorbar(cs, ticks=MESH_bounds)
f.tight_layout(pad=3.0)
plt.ylabel('y (1/2 km)')
plt.xlabel('x (1/2 km)')
plt.xlim([0,60])
plt.xticks([0,10,20,30,40,50,60])
plt.yticks([0,10,20,30,40,50,60])
plt.title(f'True MESH  #{number} (mm)')

plt.subplot(122)
cs = plt.contourf(img_pred,levels=MESH_bounds,colors=MESH_colors, extend='both', shrink=0.5, origin=None)   
plt.colorbar(cs, ticks=MESH_bounds)
f.tight_layout(pad=3.0)
plt.ylabel('y (1/2 km)')
plt.xlabel('x (1/2 km)')
plt.xlim([0,60])
plt.xticks([0,10,20,30,40,50,60])
plt.yticks([0,10,20,30,40,50,60])
plt.title(f'Predicted MESH with MSE #{number} (mm)')
st.pyplot(f, size = 100)



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
    delta = metrics.delta(A, B)
    G = metrics.G(A, B, mindists_AB, mindists_BA, beta)
    zhulak = metrics.zhulak(A, B, mindists_AB)
    medFA = np.mean(mindists_BA)
    medMiss = np.mean(mindists_AB)
    #podVilage, farVilage = metrics.neighborhood_stats_by_image(A, B, 10)
else:

    far = 0
    pod = 0
    hausdorf_distance = -99
    phdk_distance = -99
    gbeta = -99
    delta = -99
    G = -99
    zhulak = 60

metrics_dict = {'MSE': mse, 'FAR': [far], 'POD': [pod], 'Hausdorff': [hausdorf_distance], 'PHDK': [phdk_distance], 'Gbeta': [gbeta], 'G': [G], 'delta': [delta]}

metrics_dict_2 = {'zhulak': [zhulak], 'medFA': [medFA], 'medMiss': [medMiss]}

# loss_1 = mse + c1*hausdorf_distance + c2*phdk_distance + c3*-1*gbeta + c4*delta + c5*G + c6*zhulak + c7*medFA + c8*medMiss
# loss_2 = mse + (far-pod)*mse*c0 + c1*hausdorf_distance + c2*phdk_distance + c3*gbeta + c4*delta + c5*G + c6*zhulak + c7*medFA + c8*medMiss
# loss_3 = mse + mse*c1*hausdorf_distance + c2*mse*phdk_distance - c3*gbeta*mse + c4*delta*mse + c6*zhulak + c7*medFA + c8*medMiss

loss_1 = mse + mse*cHaus*hausdorf_distance
loss_2 = mse + mse*phdk_distance
loss_3 = mse + mse*cGbeta*gbeta*-1
loss_4 = mse + mse*cDelta*delta
loss_5 = mse + (far-pod)*mse*cCSI + mse*cHaus*hausdorf_distance + mse*cPHDK*phdk_distance - mse*cGbeta*gbeta + mse*cDelta*delta + cG*G + cZhu*zhulak + cFA*medFA + cMiss*medMiss

loss_dict = {'Haus': [loss_1], 'PHDK': [loss_2], 'Gbeta': [loss_3], 'Delta': [loss_4], 'All': [loss_5]}
# coeficients_dict = {'c0': [c0], 'c1': [c1], 'c2': [c2], 'c3': [c3], 'c4': [c4], 'c5': [c5], 'c6': [c6], 'c7': [c7], 'c8': [c8]}
coefficients_dict = {'cCSI': [cCSI], 'cHaus': [cHaus], 'cPHDK': [cPHDK], 'cGbeta': [cGbeta], 'cDelta': [cDelta], 'cG': [cG], 'cZhu': [cZhu], 'cFA': [cFA], 'cMiss': [cMiss]}

df = pd.DataFrame(metrics_dict)
df2 = pd.DataFrame(metrics_dict_2)
df3 = pd.DataFrame(loss_dict)
df4 = pd.DataFrame(coefficients_dict)
st.table(df3)
st.table(df)
st.table(df2)

st.table(df4)

