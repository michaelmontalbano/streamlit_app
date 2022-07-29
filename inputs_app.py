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
#, shear_colors, shear_bounds, ref_colors, ref_bounds, wind_colors, wind_bounds
import metrics

# 2: loss-mse_dataset-shave_L2-0.01_drop-0.1_junct-Add_filters-64f128_act-lrelu_cut-30_transpose-1_gain-0.0_bias-0.0_init-normal_variant-unetpp_block-vanilla_exp_index-7_kernel-3_out-act-relu_results.pkl


datasets = ['all','severe','sig-severe']
st.title('Model Prediction Metrics')
number = st.number_input("Pick a sample number (0-939)", 0, 939)
dataset = st.selectbox("Pick a dataset", datasets)
multi_fields = ['MergedReflectivityQCComposite_Max_30min','MergedLLShear_Max_30min','MergedMLShear_Max_30min','Reflectivity_0C_Max_30min','MESH_Max_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
shave_degrees = ['01.00','02.00','03.00','04.00','05.00','06.00','07.00','08.00','09.00','10.00','11.00','12.00','13.00','14.00','15.00','16.00','17.00','18.00','19.00','20.00']
shave_fields=multi_fields+NSE_fields+shave_degrees

# convert coefficients to list of comma separated floats
x_test =  load('x_test.npy')
y_true = load('data/y_true.npy').squeeze()

indices = range(len(y_true))
if dataset == 'severe':
    # keep only images where each image in y_true has max above 25
    indices = [x for x,y in zip(range(len(y_true)),y_true) if np.max(y) > 25]
if dataset == 'sig-severe':
    indices = [x for x,y in zip(range(len(y_true)),y_true) if np.max(y) > 50]
y_true = y_true[indices,:,:]
x_true = x_test[indices,:,:,:]



input = x_true[number]

field_index = st.number_input("Pick a field", 0, len(multi_fields)-1)

field = input[field_index]

# get figure name it f
f = plt.figure(figsize=(10,10))
plt.imshow(field.squeeze())
plt.colorbar()
st.pyplot(f, size = 100)

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
#


