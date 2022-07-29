MESH_colors = ['#aaaaaa','#00ffff','#0080ff','#0000ff','#007f00','#00bf00','#00ff00','#ffff00','#bfbf00','#ff9900','#ff0000','#bf0000','#7f0000','#ff1fff']
MESH_bounds = [9.525,15.875,22.225,28.575,34.925,41.275,47.625,53.975,60.325,65,70,75,80,85]

NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km']
multi_fields = ['MergedReflectivityQCComposite_Max_30min','MergedLLShear_Max_30min','MergedMLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min']
products = NSE_fields + multi_fields

shear_colors = ['#202020','#808080','#4d4d00','#636300','#bbbb00','#dddd00','#ffff00','#770000','#990000','#bb0000','#dd0000','#ff0000','#ffcccc']
shear_bounds = [0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]

MESH_colors = ['#aaaaaa','#00ffff','#0080ff','#0000ff','#007f00','#00bf00','#00ff00','#ffff00','#bfbf00','#ff9900','#ff0000','#bf0000','#7f0000','#ff1fff']
MESH_bounds = [9.525,15.875,22.225,28.575,34.925,41.275,47.625,53.975,60.325,65,70,75,80,85]

uwind_r = ['0x00','0x00','0x00','0x00','0x00','0xbf','0xff','0xff','0xff','0xbf','0x7f','0xff']
uwind_g = ['0x80','0x00','0x7f','0xbf','0xff','0xff','0xff','0xbf','0x99','0x00','0x00','0x33']
uwind_b = ['0xff','0xff','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0xff']

ref_r = [0,115,120,148,2,17,199,184,199,199,153,196,122,199]
ref_g = [0,98,120,164,199,121,196,143,113,0,0,0,69,199]
ref_b = [0,130,120,148,2,1,2,0,0,0,16,199,161,199]

ref_bounds = [-10,10,13,18,28,33,38,43,48,53,63,68,73,77]
wind_bounds = [-30,-25,-20,-15,-10,-5,-1,1,5,10,15,20,25,30]

wind_colors = ['#0080ff', '#0000ff', '#007f00', '#00bf00', '#00ff00', '#bfff00', '#ffff00', '#ffbf00', '#ff9900', '#bf0000', '#7f0000', '#ff33ff']
ref_colors = ['#000000', '#736282', '#787878', '#94a494', '#02c702', '#117901', '#c7c402', '#b88f00', '#c77100', '#c70000', '#990010', '#c400c7']
# for idx, color in enumerate(uwind_r):
#     wind_colors.append(str('#%02x%02x%02x' % (int(color,16),int(uwind_g[idx],16),int(uwind_b[idx],16))))
#     ref_colors.append(str('#%02x%02x%02x' % (ref_r[idx],ref_g[idx],ref_b[idx])))
