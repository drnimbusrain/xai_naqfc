import xarray as xr
import pandas as pd
import numpy as np

#import NOAA-ARL MONET
import monet
import monetio

#import ML models and XAI packages
import xgboost as xgb
import shap

#import plotting matplotlib
import matplotlib.pyplot as plt

# import the training and metrics from Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from functools import reduce

#month='May'
#month='June'
#month='July'
#month='August'
month='September'

#Read the model input datasets (met and emissions inputs) and outputs (ozone)
mei_fnames='../ozone_exceedance_data/raw_naqfc_24hr_10year/subset/'+month+'/aqm.*.t12z.metcro2d.ncf'
#aei_fnames='ozone_exceedance_data/raw_naqfc/emis_mole_all_202108*_AQF5X_nobeis_2016fh_16j.ncf'
#bei_fnames='ozone_exceedance_data/raw_naqfc_24hr/aqm.202108*.t12z.b3gt2.ncf'
#fei_fnames='ozone_exceedance_data/raw_naqfc/aqm.202108*.t12z.fireemis.ncf'
ozo_fnames='../ozone_exceedance_data/raw_naqfc_24hr_10year/subset/'+month+'/aqm.*.t12z.aconc.ncf'

#set lat/lon max and lat/lon min to subset the CONUS area of interest
#region='northeast'
#lonmax = -66.8628
#lonmin = -73.7272
#latmax =  47.4550
#latmin =  40.9509
#region='mid-atlantic'
#lonmax = -74.8526
#lonmin = -83.6753
#latmax =  42.5167
#latmin =  36.5427
#region='southeast'
#lonmax = -75.4129
#lonmin = -91.6589
#latmax =  39.1439
#latmin =  24.3959
#region='upper-midwest'
#lonmax = -80.5188
#lonmin = -97.2304
#latmax =  49.3877
#latmin =  36.9894
#region='south'
#lonmax = -88.7421
#lonmin = -109.0489
#latmax =  37.0015
#latmin =  25.8419
#region='central'
#lonmax = -89.1005
#lonmin = -104.0543
#latmax =  43.5008
#latmin =  35.9958
#region='upper-great-plains'
#lonmax = -96.438
#lonmin = -116.0458
#latmax =  48.9991
#latmin =  36.9949
#region='west'
#lonmax = -109.0475
#lonmin = -124.6509
#latmax =  42.0126
#latmin =  31.3325

#region='northwest'
#lonmax = -111.0471
#lonmin = -124.7305
#latmax =  49.0027
#latmin =  41.9871

#set specific regions of heavy ozone pollution west-east
#South Coast Air Basin (SoCAB) that includes urbanized portions of Los Angeles, Orange, Riverside, and San Bernardino Counties
region='SoCAB'
#lonmax = -116.676164
#lonmin = -118.913288
#latmax =  34.81774
#latmin =  33.433425

region='NYLIS'
lonmax = -72.0
lonmin = -75.0
latmax =  42.0
latmin =  40.0

#region='BWCorr'
#lonmax = -76.6122
#lonmin = -77.0369
#latmax =  39.2904
#latmin =  38.9072

#region='LMOSWest'
#lonmax = -87.40
#lonmin = -88.19
#latmax =  45.25
#latmin =  41.62

#region='LMOSEast'
#lonmax = -87.40
#lonmin = -88.19
#latmax =  45.25
#latmin =  41.62

#region='DCMetro'
#lonmax = -116.676164
#lonmin = -118.913288
#latmax =  34.81774
#latmin =  33.433425

#region='AtlantaMetro'
#lonmax = -116.676164
#lonmin = -118.913288
#latmax =  34.81774
#latmin =  33.433425

#region='COFrontRange'
#lonmax = -87.40
#lonmin = -88.19
#latmax =  45.25
#latmin =  41.62



# open the datasets using xarray and convert to dataframes
print('opening met inputs...')
met_dset_orig = monetio.models.cmaq.open_mfdataset(mei_fnames)
met_dset=met_dset_orig
if month == 'May' or month == 'June':
 met_df=met_dset[['TEMPG','TEMP2','WSPD10','WDIR10','PRSFC', 'PBL', 'QFX','HFX','RGRND','CFRAC','RSTOMI','RADYNI','RN','RC']].to_dataframe().reset_index()
else:
 met_df=met_dset[['TEMPG','TEMP2','WSPD10','WDIR10','PRSFC','Q2','PBL', 'QFX','HFX','RGRND','CFRAC','VEG','RSTOMI','RADYNI','RN','RC','SOIT1','SOIM1']].to_dataframe().reset_index()
met_df['PRECIP']=met_df['RN']+met_df['RC']
met_df.drop(columns=['x', 'y', 'z','RN','RC'], inplace=True)
met_df.query('latitude > @latmin & latitude < @latmax & longitude < @lonmax & longitude > @lonmin & RGRND > 0.0',inplace=True)

#print('opening ant. emission inputs...')
#aei_dset_orig = monetio.models.cmaq.open_mfdataset(aei_fnames)
#aei_dset=aei_dset_orig
#aei_dset=aei_dset_orig.coarsen(x=6,boundary="trim").mean().coarsen(y=6,boundary="trim").mean()

#aei_df=aei_dset[['NO','NO2','VOC_INV','CO']].to_dataframe().reset_index()
#aei_df.rename(columns={'NO': 'AE_NO', 'NO2': 'AE_NO2', 'VOC_INV': 'AE_VOC', 'CO': 'AE_CO'}, inplace=True)
#aei_df['AE_NOX']=aei_df['AE_NO']+aei_df['AE_NO2']
#aei_df.drop(columns=['x', 'y', 'z','AE_NO','AE_NO2'], inplace=True)
#aei_df.query('latitude > @latmin & latitude < @latmax & longitude < @lonmax & longitude > @lonmin',inplace=True)
#print(aei_df)

#print('opening bio. emission inputs...')
#bei_dset_orig = monetio.models.cmaq.open_mfdataset(bei_fnames)
#bei_dset=bei_dset_orig
#bei_dset=bei_dset_orig.coarsen(x=6,boundary="trim").mean().coarsen(y=6,boundary="trim").mean()

#bei_df   = bei_dset[['ISOP','OLE','PAR','MEOH','APIN','TERP','ETH','ETOH','ACET','ALDX','IOLE','FORM','ALD2','ETHA','KET','NO']].to_dataframe().reset_index()
#bei_df.rename(columns={'ISOP': 'BE_ISOP', 'OLE': 'BE_OLE', 'PAR': 'BE_PAR', 'MEOH': 'BE_MEOH','APIN': 'BE_APIN','TERP': 'BE_TERP','ETH': 'BE_ETH','ETOH': 'BE_ETOH','ACET': 'BE_ACET','ALDX': 'BE_ALDX','IOLE': 'BE_IOLE','FORM': 'BE_FORM','ALD2': 'BE_ALD2','ETHA': 'BE_ETHA','KET': 'BE_KET','NO': 'BE_NO'}, inplace=True)
#bei_df['BE_VOC']=bei_df['BE_ISOP']+bei_df['BE_OLE']+bei_df['BE_PAR']+bei_df['BE_MEOH']+bei_df['BE_APIN']+bei_df['BE_TERP']+bei_df['BE_ETH']+bei_df['BE_ETOH']+bei_df['BE_ACET']+bei_df['BE_ALDX']+bei_df['BE_IOLE']+bei_df['BE_FORM']+bei_df['BE_ALD2']+bei_df['BE_ETHA']+bei_df['BE_KET']
#bei_df.drop(columns=['x', 'y', 'z','BE_ISOP','BE_OLE','BE_PAR','BE_MEOH','BE_APIN','BE_TERP','BE_ETH','BE_ETOH','BE_ACET','BE_ALDX','BE_IOLE','BE_FORM','BE_ALD2','BE_ETHA','BE_KET'], inplace=True)
#bei_df.query('latitude > @latmin & latitude < @latmax & longitude < @lonmax & longitude > @lonmin',inplace=True)
##print(bei_df)
#
#print('opening fire emission inputs...')
#fei_dset_orig = monetio.models.cmaq.open_mfdataset(fei_fnames)
#fei_dset=fei_dset_orig
##fei_dset=fei_dset_orig.coarsen(x=6,boundary="trim").mean().coarsen(y=6,boundary="trim").mean()
#
#fei_df   = fei_dset[['NO','NO2','CO','ISOP','OLE','PAR','MEOH','TERP','ETH','ETOH','ACET','ALDX','IOLE','FORM','ALD2','ETHA','KET']].to_dataframe().reset_index()
#fei_df.rename(columns={'NO': 'FE_NO', 'NO2': 'FE_NO2', 'CO': 'FE_CO','ISOP': 'FE_ISOP', 'OLE': 'FE_OLE', 'PAR': 'FE_PAR', 'MEOH': 'FE_MEOH','TERP': 'FE_TERP','ETH': 'FE_ETH','ETOH': 'FE_ETOH','ACET': 'FE_ACET','ALDX': 'FE_ALDX','IOLE': 'FE_IOLE','FORM': 'FE_FORM','ALD2': 'FE_ALD2','ETHA': 'FE_ETHA','KET': 'FE_KET'}, inplace=True)
#fei_df['FE_VOC']=fei_df['FE_ISOP']+fei_df['FE_OLE']+fei_df['FE_PAR']+fei_df['FE_MEOH']+fei_df['FE_TERP']+fei_df['FE_ETH']+fei_df['FE_ETOH']+fei_df['FE_ACET']+fei_df['FE_ALDX']+fei_df['FE_IOLE']+fei_df['FE_FORM']+fei_df['FE_ALD2']+fei_df['FE_ETHA']+fei_df['FE_KET']
#fei_df['FE_NOX']=fei_df['FE_NO']+fei_df['FE_NO2']
#fei_df.drop(columns=['x', 'y', 'z','FE_ISOP','FE_OLE','FE_PAR','FE_MEOH','FE_TERP','FE_ETH','FE_ETOH','FE_ACET','FE_ALDX','FE_IOLE','FE_FORM','FE_ALD2','FE_ETHA','FE_KET','FE_NO','FE_NO2'], inplace=True)
#fei_df.query('latitude > @latmin & latitude < @latmax & longitude < @lonmax & longitude > @lonmin',inplace=True)
#print(fei_df)

print('opening ozone outputs and set target ozone value...')
ozo_dset_orig = monetio.models.cmaq.open_mfdataset(ozo_fnames)
ozo_dset=ozo_dset_orig
#ozo_dset=ozo_dset_orig.coarsen(x=6,boundary="trim").mean().coarsen(y=6,boundary="trim").mean()
ozo_df=ozo_dset['O3'].to_dataframe().reset_index()
ozo_df.query('latitude > @latmin & latitude < @latmax & longitude < @lonmax & longitude > @lonmin',inplace=True)

print('calculate 8hr rolling ozone average...')
ozo_df['O3_8hr']=ozo_df['O3'].rolling(8).mean()
#ozo_df['target']=np.where(ozo_df['O3']>= 70.0, 1, 0)
ozo_df['target']=np.where(ozo_df['O3_8hr']>= 70.0, 1, 0)
ozo_df.drop(columns=['x', 'y', 'z','O3','O3_8hr'], inplace=True)
print(ozo_df['target'].value_counts())

# Merging dataframes  on time, lat, lons
print('merging dataframes...')
#data_frames = [met_df, aei_df, bei_df, fei_df, ozo_df]
data_frames = [met_df, ozo_df]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time','latitude','longitude'],how='outer'), data_frames)

df = df_merged.drop(columns=['time', 'latitude', 'longitude'])
df = df.dropna()

# Show the first five rows
print(df.head())

# Set up the data for modelling
y=df['target'].to_frame() # define Y
X=df[df.columns.difference(['target'])] # define X
print('create train and test...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # create train and test

# build model - Xgboost
print('build model...')
xgb_mod=xgb.XGBClassifier(random_state=42,gpu_id=0) # build classifier
xgb_mod=xgb_mod.fit(X_train,y_train.values.ravel())

# make prediction and check model accuracy
print('make prediction...')
y_pred = xgb_mod.predict(X_test)

# Performance
print('check performance...')
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Shapely XAI methods  --->
# Generate the Tree explainer and SHAP values
print('generate tree explainer and shap values...')
explainer = shap.TreeExplainer(xgb_mod)
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value

last_explanation=shap_values[:,0].size - 1

############## visualizations #############
print('shap visualizations...')
# Generate summary dot plot
shap.summary_plot(shap_values, X,title="SHAP summary plot",show=False)
plt.savefig("Figure_1_shap_value_10year_"+region+"_"+month+".png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

## Generate summary bar plot
shap.summary_plot(shap_values, X,plot_type="bar",show=False)
plt.savefig("Figure_2_mean_shap_value_10year_"+region+"_"+month+".png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

## Generate specific dependence plots
shap.dependence_plot("PBL", shap_values, X, interaction_index="TEMP2", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_PBL_TEMP2.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

shap.dependence_plot("TEMP2", shap_values, X, interaction_index="PBL", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_TEMP2_PBL.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

shap.dependence_plot("RGRND", shap_values, X, interaction_index="CFRAC", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_RGRND_CFRAC.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

shap.dependence_plot("CFRAC", shap_values, X, interaction_index="RGRND", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_CFRAC_RGRND.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

shap.dependence_plot("PRSFC", shap_values, X, interaction_index="WSPD10", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_PRSFC_WSPD10.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

shap.dependence_plot("WSPD10", shap_values, X, interaction_index="PRSFC", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_WSPD10_PRSFC.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

if month == 'July' or month == 'August' or month == 'September':
 shap.dependence_plot("Q2", shap_values, X, interaction_index=None, alpha=0.1, show=False)
 plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_Q2.png", format='png', dpi='figure', bbox_inches='tight')
 plt.close()

#shap.dependence_plot("BE_VOC", shap_values, X, interaction_index="TEMP2", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_BE_VOC_TEMP2.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("TEMP2", shap_values, X, interaction_index="BE_VOC", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_TEMP2_BE_VOC.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("BE_VOC", shap_values, X, interaction_index="RGRND", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_BE_VOC_RGRND.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("BE_VOC", shap_values, X, interaction_index="LAI", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_BE_VOC_LAI.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("LAI", shap_values, X, interaction_index="BE_VOC", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_LAI_BE_VOC.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("RGRND", shap_values, X, interaction_index="BE_VOC", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_RGRND_BE_VOC.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("BE_NO", shap_values, X, interaction_index="PRECIP", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_BE_NO_PRECIP.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("PRECIP", shap_values, X, interaction_index="BE_NO", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_PRECIP_BE_NO.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()

shap.dependence_plot("WDIR10", shap_values, X, interaction_index="WSPD10", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_WDIR10_WSPD10.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

shap.dependence_plot("WDIR10", shap_values, X, interaction_index="TEMP2", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_WDIR10_TEMP2.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

shap.dependence_plot("WSPD10", shap_values, X, interaction_index="WDIR10", alpha=0.1, show=False)
plt.savefig("Figure_3_dependence_plot_10year_"+region+"_"+month+"_WSPD10_WDIR10.png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

#shap.dependence_plot("AE_NOX", shap_values, X, interaction_index="AE_VOC", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_AE_NOX_AE_VOC.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("AE_VOC", shap_values, X, interaction_index="AE_NOX", alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_AE_VOC_AE_NOX.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.dependence_plot("AE_CO", shap_values, X, interaction_index=None, alpha=0.1, show=False)
#plt.savefig("Figure_3_dependence_plot_"+region+"_AE_CO.png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()

### Generate multiple dependence plots
#for name in X_train.columns:
#     shap.dependence_plot(name, shap_values, X, show=False)
#     plt.savefig("Figure_4_dependence_plot_"+region+"_"+name+".png", format='png', dpi='figure', bbox_inches='tight')
#     plt.close()
#
## Generate waterfall plot
shap.plots._waterfall.waterfall_legacy(expected_value,shap_values[0], feature_names=X.columns, max_display=20, show=False)
plt.savefig("Figure_5_waterfall_legacy_first_10year_"+region+"_"+month+".png", format='png', dpi='figure', bbox_inches='tight')
plt.close()
#
shap.plots._waterfall.waterfall_legacy(expected_value,shap_values[last_explanation], feature_names=X.columns, max_display=20, show=False)
plt.savefig("Figure_5_waterfall_legacy_last_10year_"+region+"_"+month+".png", format='png', dpi='figure', bbox_inches='tight')
plt.close()
#
## Generate force plot - Single row
#shap.force_plot(explainer.expected_value, shap_values[0], feature_names=X.columns, show=False)
#plt.savefig("Figure_6_force_single_first_"+region+".png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
#shap.force_plot(explainer.expected_value, shap_values[last_explanation], feature_names=X.columns, show=False)
#plt.savefig("Figure_6_force_single_last_"+region+".png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
## Generate force plot - Multiple rows
#shap.force_plot(explainer.expected_value, shap_values[:100,:], feature_names=X.columns, show=False)
#plt.savefig("Figure_6_force_multiple_100_"+region+".png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
#
## Generate Decision plot
shap.decision_plot(expected_value, shap_values[0],  link='logit', feature_names=(X.columns.tolist()),ignore_warnings=True, show=False)
plt.savefig("Figure_7_decision_plot_first_10year_"+region+"_"+month+".png", format='png', dpi='figure', bbox_inches='tight')
plt.close()
#
shap.decision_plot(expected_value, shap_values[last_explanation], link='logit', feature_names=(X.columns.tolist()),ignore_warnings=True, show=False)
plt.savefig("Figure_7_decision_plot_last_10year_"+region+"_"+month+".png", format='png', dpi='figure', bbox_inches='tight')
plt.close()
#
shap.decision_plot(expected_value, shap_values[:100,:],  link='logit', feature_names=(X.columns.tolist()),ignore_warnings=True, show=False)
plt.savefig("Figure_7_decision_plot_multiple_100_10year_"+region+"_"+month+".png", format='png', dpi='figure', bbox_inches='tight')
plt.close()

#shap.plots.heatmap(shap_values[1:100])
#plt.savefig("Figure_8_heat_map_"+region+".png", format='png', dpi='figure', bbox_inches='tight')
#plt.close()
