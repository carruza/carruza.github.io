---
layout: post
title:  "Tanzania Water Pumps"
date:   2020-05-04 10:03:38 -0300
categories: jekyll update
---
# Pump it Up: Data Mining the Water Table

*Using data from Taarifa and the Tanzanian Ministry of Water, we predict which pumps are functional, which need some repairs, and which don't work at all. Here, we predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania. This is an intermediate-level practice competition.*

https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/


```python
import pandas as pd
import numpy as np
import seaborn as sns

%matplotlib inline
```


```python
pd.set_option('display.max_columns',999)
```


```python
train = pd.read_csv('TrainingValues.csv', index_col = 0)
labels = pd.read_csv('TrainingLabels.csv', index_col = 0)
```


```python
train.shape
```




    (59400, 39)




```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>date_recorded</th>
      <th>funder</th>
      <th>gps_height</th>
      <th>installer</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>wpt_name</th>
      <th>num_private</th>
      <th>basin</th>
      <th>subvillage</th>
      <th>region</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>lga</th>
      <th>ward</th>
      <th>population</th>
      <th>public_meeting</th>
      <th>recorded_by</th>
      <th>scheme_management</th>
      <th>scheme_name</th>
      <th>permit</th>
      <th>construction_year</th>
      <th>extraction_type</th>
      <th>extraction_type_group</th>
      <th>extraction_type_class</th>
      <th>management</th>
      <th>management_group</th>
      <th>payment</th>
      <th>payment_type</th>
      <th>water_quality</th>
      <th>quality_group</th>
      <th>quantity</th>
      <th>quantity_group</th>
      <th>source</th>
      <th>source_type</th>
      <th>source_class</th>
      <th>waterpoint_type</th>
      <th>waterpoint_type_group</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69572</th>
      <td>6000.0</td>
      <td>2011-03-14</td>
      <td>Roman</td>
      <td>1390</td>
      <td>Roman</td>
      <td>34.938093</td>
      <td>-9.856322</td>
      <td>none</td>
      <td>0</td>
      <td>Lake Nyasa</td>
      <td>Mnyusi B</td>
      <td>Iringa</td>
      <td>11</td>
      <td>5</td>
      <td>Ludewa</td>
      <td>Mundindi</td>
      <td>109</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Roman</td>
      <td>False</td>
      <td>1999</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay annually</td>
      <td>annually</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>8776</th>
      <td>0.0</td>
      <td>2013-03-06</td>
      <td>Grumeti</td>
      <td>1399</td>
      <td>GRUMETI</td>
      <td>34.698766</td>
      <td>-2.147466</td>
      <td>Zahanati</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Nyamara</td>
      <td>Mara</td>
      <td>20</td>
      <td>2</td>
      <td>Serengeti</td>
      <td>Natta</td>
      <td>280</td>
      <td>NaN</td>
      <td>GeoData Consultants Ltd</td>
      <td>Other</td>
      <td>NaN</td>
      <td>True</td>
      <td>2010</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>wug</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>34310</th>
      <td>25.0</td>
      <td>2013-02-25</td>
      <td>Lottery Club</td>
      <td>686</td>
      <td>World vision</td>
      <td>37.460664</td>
      <td>-3.821329</td>
      <td>Kwa Mahundi</td>
      <td>0</td>
      <td>Pangani</td>
      <td>Majengo</td>
      <td>Manyara</td>
      <td>21</td>
      <td>4</td>
      <td>Simanjiro</td>
      <td>Ngorika</td>
      <td>250</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Nyumba ya mungu pipe scheme</td>
      <td>True</td>
      <td>2009</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay per bucket</td>
      <td>per bucket</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>dam</td>
      <td>dam</td>
      <td>surface</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>67743</th>
      <td>0.0</td>
      <td>2013-01-28</td>
      <td>Unicef</td>
      <td>263</td>
      <td>UNICEF</td>
      <td>38.486161</td>
      <td>-11.155298</td>
      <td>Zahanati Ya Nanyumbu</td>
      <td>0</td>
      <td>Ruvuma / Southern Coast</td>
      <td>Mahakamani</td>
      <td>Mtwara</td>
      <td>90</td>
      <td>63</td>
      <td>Nanyumbu</td>
      <td>Nanyumbu</td>
      <td>58</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>True</td>
      <td>1986</td>
      <td>submersible</td>
      <td>submersible</td>
      <td>submersible</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>dry</td>
      <td>dry</td>
      <td>machine dbh</td>
      <td>borehole</td>
      <td>groundwater</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>19728</th>
      <td>0.0</td>
      <td>2011-07-13</td>
      <td>Action In A</td>
      <td>0</td>
      <td>Artisan</td>
      <td>31.130847</td>
      <td>-1.825359</td>
      <td>Shuleni</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Kyanyamisa</td>
      <td>Kagera</td>
      <td>18</td>
      <td>1</td>
      <td>Karagwe</td>
      <td>Nyakasimbi</td>
      <td>0</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>0</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>other</td>
      <td>other</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>seasonal</td>
      <td>seasonal</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.columns.tolist()
```




    ['amount_tsh',
     'date_recorded',
     'funder',
     'gps_height',
     'installer',
     'longitude',
     'latitude',
     'wpt_name',
     'num_private',
     'basin',
     'subvillage',
     'region',
     'region_code',
     'district_code',
     'lga',
     'ward',
     'population',
     'public_meeting',
     'recorded_by',
     'scheme_management',
     'scheme_name',
     'permit',
     'construction_year',
     'extraction_type',
     'extraction_type_group',
     'extraction_type_class',
     'management',
     'management_group',
     'payment',
     'payment_type',
     'water_quality',
     'quality_group',
     'quantity',
     'quantity_group',
     'source',
     'source_type',
     'source_class',
     'waterpoint_type',
     'waterpoint_type_group']



We see a high number of unique values in some categorical variables, such as waterpoint names (wpt_name), scheme names (scheme_name) and subvillage.


```python
merged = train.join(labels)
```


```python
merged.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>gps_height</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>num_private</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>population</th>
      <th>construction_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>5.940000e+04</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>317.650385</td>
      <td>668.297239</td>
      <td>34.077427</td>
      <td>-5.706033e+00</td>
      <td>0.474141</td>
      <td>15.297003</td>
      <td>5.629747</td>
      <td>179.909983</td>
      <td>1300.652475</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2997.574558</td>
      <td>693.116350</td>
      <td>6.567432</td>
      <td>2.946019e+00</td>
      <td>12.236230</td>
      <td>17.587406</td>
      <td>9.633649</td>
      <td>471.482176</td>
      <td>951.620547</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-90.000000</td>
      <td>0.000000</td>
      <td>-1.164944e+01</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>33.090347</td>
      <td>-8.540621e+00</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>369.000000</td>
      <td>34.908743</td>
      <td>-5.021597e+00</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>1986.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>20.000000</td>
      <td>1319.250000</td>
      <td>37.178387</td>
      <td>-3.326156e+00</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>5.000000</td>
      <td>215.000000</td>
      <td>2004.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>350000.000000</td>
      <td>2770.000000</td>
      <td>40.345193</td>
      <td>-2.000000e-08</td>
      <td>1776.000000</td>
      <td>99.000000</td>
      <td>80.000000</td>
      <td>30500.000000</td>
      <td>2013.000000</td>
    </tr>
  </tbody>
</table>
</div>



Some observations are missing longitudinal data


```python
merged[merged['longitude'] == 0].shape
```




    (1812, 40)



Classes are slightly umbalanced, but we believe over/undersampling is not necessary for predicting.


```python

```


```python
print(merged['status_group'].value_counts().sort_values())
merged['status_group'].value_counts().sort_values().plot(kind = 'barh')
```

    functional needs repair     4317
    non functional             22824
    functional                 32259
    Name: status_group, dtype: int64





    <matplotlib.axes._subplots.AxesSubplot at 0x1225ba3d0>




![png]({{"/assets/images/output_15_2.png"}})



```python
print('Unique values for each categorical variable:')
for x in train.select_dtypes(include = 'object'):

    print(x, train[x].nunique(), sep = ' -> ')
```

    Unique values for each categorical variable:
    date_recorded -> 356
    funder -> 1897
    installer -> 2145
    wpt_name -> 37400
    basin -> 9
    subvillage -> 19287
    region -> 21
    lga -> 125
    ward -> 2092
    public_meeting -> 2
    recorded_by -> 1
    scheme_management -> 12
    scheme_name -> 2696
    permit -> 2
    extraction_type -> 18
    extraction_type_group -> 13
    extraction_type_class -> 7
    management -> 12
    management_group -> 5
    payment -> 7
    payment_type -> 7
    water_quality -> 8
    quality_group -> 6
    quantity -> 5
    quantity_group -> 5
    source -> 10
    source_type -> 7
    source_class -> 3
    waterpoint_type -> 7
    waterpoint_type_group -> 6


### Exploratory data analysis

We'll narrow the exploratory analysis by considering waterpumps in need of repair as functioning waterpumps.


```python
merged['functional_or_not'] = merged['status_group'].apply(lambda x: 'functional' if x == 'functional' else 'not functional')
```


```python
import geopandas
import matplotlib.pyplot as plt
```


```python
location = merged.reset_index()[['id', 'region', 'latitude', 'longitude','status_group','functional_or_not']]
```


```python
to_drop = location[location['longitude'] == 0].index
```


```python
location.drop(to_drop, inplace = True)
```


```python
gdf = geopandas.GeoDataFrame(
    location, geometry=geopandas.points_from_xy(location.longitude, location.latitude))
```


```python
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
```


```python
sns.set(rc={'figure.figsize':(11.7,8.27)}, palette = 'deep')
```


```python
world["tanz"] = world.name.apply(lambda x: x if x == 'Tanzania' else np.nan)
ax = world[world['continent'] == 'Africa'].plot(color="white", edgecolor = 'black')
world.dropna().plot(ax=ax, column="name")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12baf5390>




![png]({{"/assets/images/output_27_1.png"}})



```python
ax = world[world.name == 'Tanzania'].plot(
    color='white', edgecolor='black')

gdf.plot(ax=ax, color='black', markersize = 0.5)


plt.title('Tanzania')
plt.show()
```


![png]({{"/assets/images/output_28_0.png"}})


Functioning pumps in blue, those not functioning are in red.


```python
ax = world[world.name == 'Tanzania'].plot(
    color='white', edgecolor='black')

gdf[gdf['functional_or_not'] == 'functional'].plot(ax=ax, color='blue', markersize = 0.5)
gdf[gdf['functional_or_not'] == 'not functional'].plot(ax=ax, color='red', markersize = 0.5)


plt.title('Tanzania')
plt.show()
```


![png]({{"/assets/images/output_30_0.png"}})


#### Waterpump building has increased over the years, peaking in 2010.


```python
g = sns.countplot(x = 'construction_year', color = 'blue', saturation = 0.1, data = merged.query('construction_year != 0'))
g.tick_params(labelsize = 10, labelrotation=90)
```


![png]({{"/assets/images/output_32_0.png"}})


Years since built appears to be a factor for waterpump functioning, as newer pumps are the ones in better shape.


```python
p = sns.countplot(x = 'construction_year', hue = 'functional_or_not', data = merged.query('construction_year != 0'))
p.tick_params(labelsize = 10, labelrotation=90)
```


![png]({{"/assets/images/output_34_0.png"}})



```python
new = merged.groupby(['construction_year','functional_or_not'])['amount_tsh'].mean()[3:].reset_index()
```

There also appears to be a relationship between the amount water available to waterpoint and it's functionality. Waterpumps with higher capacity tend to break less.


```python
sns.lineplot(x = 'construction_year', y = 'amount_tsh', hue = 'functional_or_not', data = new)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x123d156d0>




![png]({{"/assets/images/output_37_1.png"}})



```python
new2 = merged.groupby(['construction_year','functional_or_not'])['gps_height'].mean()[3:].reset_index()
```

Also waterpoints with larger well depth show more functioning. May be higher depth requires higher waterpump quality.


```python
sns.lineplot(x = 'construction_year', y = 'gps_height', hue = 'functional_or_not', data = new2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x105c51650>




![png]({{"/assets/images/output_40_1.png"}})



```python
print('Construction year == 0:',merged[merged['construction_year'] == 0].shape[0])
```

    Construction year == 0: 20709


The Department of Water Engineer (DWE) is the main waterpump installer in Tanzania.


```python
merged['installer'].value_counts().head(15)
```




    DWE                   17402
    Government             1825
    RWE                    1206
    Commu                  1060
    DANIDA                 1050
    KKKT                    898
    Hesawa                  840
    0                       777
    TCRS                    707
    Central government      622
    CES                     610
    Community               553
    DANID                   552
    District Council        551
    HESAWA                  539
    Name: installer, dtype: int64




```python
merged['installer'].value_counts().head(10).plot(kind = 'bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11cb3af90>




![png]({{"/assets/images/output_44_1.png"}})



```python
big_installers = merged['installer'].value_counts().index[:10].tolist()
merged['installer2'] = merged['installer'].apply(lambda x: x if x in big_installers else 'Other')
```


```python
inst = merged.groupby(['installer2','functional_or_not']).size().reset_index()
```


```python
inst.rename(columns = {0:'count'}, inplace = True)
```


```python
inst.drop(inst.loc[inst['installer2'] == 'Other'].index, inplace = True)
```

Data shows that government installed waterpumps tend to break.


```python
sns.barplot(x = 'installer2', y = 'count', hue = 'functional_or_not' , data = inst)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11e376fd0>




![png]({{"/assets/images/output_50_1.png"}})


The Government of Tanzania is the main waterpoint infrastructure funder.


```python
merged['funder'].value_counts().head(15)
```




    Government Of Tanzania    9084
    Danida                    3114
    Hesawa                    2202
    Rwssp                     1374
    World Bank                1349
    Kkkt                      1287
    World Vision              1246
    Unicef                    1057
    Tasaf                      877
    District Council           843
    Dhv                        829
    Private Individual         826
    Dwsp                       811
    0                          777
    Norad                      765
    Name: funder, dtype: int64




```python
merged['funder'].value_counts().head(10).plot(kind = 'bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x120bc5190>




![png]({{"/assets/images/output_53_1.png"}})


A cause for non-functioning waterpumps is waterpoints being dry.


```python
sns.factorplot(x="quantity", hue = 'functional_or_not', kind = 'count',
            data=merged)
```

    //anaconda3/lib/python3.7/site-packages/seaborn/categorical.py:3669: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)





    <seaborn.axisgrid.FacetGrid at 0x120f82ed0>




![png]({{"/assets/images/output_55_2.png"}})


Population data shows a non-trivial quantity of outliers. 90% of waterpumps provide water for less than 500-individual groups.


```python
merged['population'].quantile([0.25,0.5,0.75, 0.9,0.99])
```




    0.25       0.0
    0.50      25.0
    0.75     215.0
    0.90     453.0
    0.99    2000.0
    Name: population, dtype: float64




```python
sns.boxplot(y = 'population', data = merged)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1210a9dd0>




![png]({{"/assets/images/output_58_1.png"}})



```python
sns.boxplot(y = 'population', data = merged.query('population < 300'))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x122bfe0d0>




![png]({{"/assets/images/output_59_1.png"}})


Most common type of waterpoint is the communal standpipe, followed by hand pump waterpoints.


```python
g = sns.countplot(x = 'waterpoint_type', data = merged)
g.tick_params(labelrotation=30)
```


![png]({{"/assets/images/output_61_0.png"}})



```python
by_wpt = merged.query('construction_year != 0').groupby(['waterpoint_type','construction_year']).size().reset_index()
```


```python
by_wpt.rename(columns = {0:'count'}, inplace = True)
```

Types of waterpoints installed throughout time.


```python
sns.lineplot(x = 'construction_year',y = 'count', hue = 'waterpoint_type', data = by_wpt)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x121021350>




![png]({{"/assets/images/output_65_1.png"}})


Next chart shows that cattle trough and multiple communal standpipes are used to provide water to large populations.


```python
merged.query('population < 500').groupby('waterpoint_type')['population'].mean().sort_values(ascending = False).plot(kind = 'bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11bf21050>




![png]({{"/assets/images/output_67_1.png"}})


Multiple communal standpipes exhibit more non-functional than functional units.


```python
sns.countplot(x = 'waterpoint_type',hue = 'functional_or_not', data = merged).tick_params(labelrotation=30)
```


![png]({{"/assets/images/output_69_0.png"}})


### Modelling and predicting


```python
train2 = train.copy()
```


```python
train2 = train2.join(labels)
```


```python
train2 = train2.query('longitude != 0')
train2 = train2.query('population < 1000')
```


```python
train_labels = train2['status_group']
```


```python
train2 = train2.iloc[:,:-1]
```


```python
train2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>date_recorded</th>
      <th>funder</th>
      <th>gps_height</th>
      <th>installer</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>wpt_name</th>
      <th>num_private</th>
      <th>basin</th>
      <th>subvillage</th>
      <th>region</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>lga</th>
      <th>ward</th>
      <th>population</th>
      <th>public_meeting</th>
      <th>recorded_by</th>
      <th>scheme_management</th>
      <th>scheme_name</th>
      <th>permit</th>
      <th>construction_year</th>
      <th>extraction_type</th>
      <th>extraction_type_group</th>
      <th>extraction_type_class</th>
      <th>management</th>
      <th>management_group</th>
      <th>payment</th>
      <th>payment_type</th>
      <th>water_quality</th>
      <th>quality_group</th>
      <th>quantity</th>
      <th>quantity_group</th>
      <th>source</th>
      <th>source_type</th>
      <th>source_class</th>
      <th>waterpoint_type</th>
      <th>waterpoint_type_group</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69572</th>
      <td>6000.0</td>
      <td>2011-03-14</td>
      <td>Roman</td>
      <td>1390</td>
      <td>Roman</td>
      <td>34.938093</td>
      <td>-9.856322</td>
      <td>none</td>
      <td>0</td>
      <td>Lake Nyasa</td>
      <td>Mnyusi B</td>
      <td>Iringa</td>
      <td>11</td>
      <td>5</td>
      <td>Ludewa</td>
      <td>Mundindi</td>
      <td>109</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Roman</td>
      <td>False</td>
      <td>1999</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay annually</td>
      <td>annually</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>8776</th>
      <td>0.0</td>
      <td>2013-03-06</td>
      <td>Grumeti</td>
      <td>1399</td>
      <td>GRUMETI</td>
      <td>34.698766</td>
      <td>-2.147466</td>
      <td>Zahanati</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Nyamara</td>
      <td>Mara</td>
      <td>20</td>
      <td>2</td>
      <td>Serengeti</td>
      <td>Natta</td>
      <td>280</td>
      <td>NaN</td>
      <td>GeoData Consultants Ltd</td>
      <td>Other</td>
      <td>NaN</td>
      <td>True</td>
      <td>2010</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>wug</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>34310</th>
      <td>25.0</td>
      <td>2013-02-25</td>
      <td>Lottery Club</td>
      <td>686</td>
      <td>World vision</td>
      <td>37.460664</td>
      <td>-3.821329</td>
      <td>Kwa Mahundi</td>
      <td>0</td>
      <td>Pangani</td>
      <td>Majengo</td>
      <td>Manyara</td>
      <td>21</td>
      <td>4</td>
      <td>Simanjiro</td>
      <td>Ngorika</td>
      <td>250</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Nyumba ya mungu pipe scheme</td>
      <td>True</td>
      <td>2009</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay per bucket</td>
      <td>per bucket</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>dam</td>
      <td>dam</td>
      <td>surface</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>67743</th>
      <td>0.0</td>
      <td>2013-01-28</td>
      <td>Unicef</td>
      <td>263</td>
      <td>UNICEF</td>
      <td>38.486161</td>
      <td>-11.155298</td>
      <td>Zahanati Ya Nanyumbu</td>
      <td>0</td>
      <td>Ruvuma / Southern Coast</td>
      <td>Mahakamani</td>
      <td>Mtwara</td>
      <td>90</td>
      <td>63</td>
      <td>Nanyumbu</td>
      <td>Nanyumbu</td>
      <td>58</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>True</td>
      <td>1986</td>
      <td>submersible</td>
      <td>submersible</td>
      <td>submersible</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>dry</td>
      <td>dry</td>
      <td>machine dbh</td>
      <td>borehole</td>
      <td>groundwater</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>19728</th>
      <td>0.0</td>
      <td>2011-07-13</td>
      <td>Action In A</td>
      <td>0</td>
      <td>Artisan</td>
      <td>31.130847</td>
      <td>-1.825359</td>
      <td>Shuleni</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Kyanyamisa</td>
      <td>Kagera</td>
      <td>18</td>
      <td>1</td>
      <td>Karagwe</td>
      <td>Nyakasimbi</td>
      <td>0</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>0</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>other</td>
      <td>other</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>seasonal</td>
      <td>seasonal</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
  </tbody>
</table>
</div>




```python
train2['district_code'] = train2['district_code'].astype('category')
train2['construction_year'] = train2['construction_year'].astype('category')
```

We drop high cardinality variables.


```python
train2.drop(['date_recorded','wpt_name', 'subvillage', 'scheme_name','ward','lga', 'num_private','scheme_management',
             'region_code','recorded_by', 'quantity_group','extraction_type',
             'extraction_type_class', 'waterpoint_type_group'], axis = 1, inplace = True)
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(train2, train_labels, test_size = 0.33, random_state = 42)
```


```python
X_train.shape
```




    (37451, 25)




```python
X_test.shape
```




    (18447, 25)



After splitting, we reduce cardinality on some columns believed to be useful.


```python
big_funders = X_train['funder'].value_counts().index[:10].tolist()
X_train['funder'] = X_train['funder'].apply(lambda x: x if x in big_funders else 'Other')
```

    //anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
big_inst = X_train['installer'].value_counts().index[:10].tolist()
X_train['installer'] = X_train['installer'].apply(lambda x: x if x in big_inst else 'Other')
```

    //anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
X_train.shape
```




    (37451, 25)




```python
X_test.shape
```




    (18447, 25)




```python
#big_funders = X_test['funder'].value_counts().index[:10].tolist()
X_test['funder'] = X_test['funder'].apply(lambda x: x if x in big_funders else 'Other')
```

    //anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
#big_inst = X_test['installer'].value_counts().index[:10].tolist()
X_test['installer'] = X_test['installer'].apply(lambda x: x if x in big_inst else 'Other')
```

    //anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
X_train.shape
```




    (37451, 25)




```python
X_test.shape
```




    (18447, 25)




```python
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
```


```python
X_train.shape
```




    (37451, 226)




```python
X_test.shape
```




    (18447, 226)




```python
from sklearn.metrics import confusion_matrix, accuracy_score
```
We first use logistic regression for predicting, settling our baseline model.

```python
from sklearn.linear_model import LogisticRegression
```


```python
clf = LogisticRegression().fit(X_train, y_train)
```

    //anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)



```python
preds = clf.predict(X_test)
```


```python
confusion_matrix(y_test, preds)
```




    array([[8179,    0, 1867],
           [ 973,    0,  282],
           [3190,    0, 3956]])




```python
accuracy_score(y_test, preds)
```




    0.6578305415514718



##### We chose Random Forest Classifier as it provides some level of feature importance.


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
from sklearn.model_selection import GridSearchCV
```
parameters = {'n_estimators':(100,150,200,250,300)}
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc, parameters)
clf.fit(X_train, y_train)
After running a GridSearch we found that the optimal number of estimators for Random Forest Classifier was 200.
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)


```python
clf = RandomForestClassifier(n_estimators = 200).fit(X_train, y_train)
```


```python
preds = clf.predict(X_test)
```


```python
confusion_matrix(y_test, preds)
```




    array([[8818,  318,  910],
           [ 629,  423,  203],
           [1431,  128, 5587]])




```python
accuracy_score(y_test, preds)
```




    0.8038163387000596




```python
features = pd.DataFrame(list(zip(X_train.columns.tolist(),clf.feature_importances_)), columns = ['feature','importance'])
```


```python
features.sort_values('importance', ascending = False).head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>longitude</td>
      <td>0.127886</td>
    </tr>
    <tr>
      <th>3</th>
      <td>latitude</td>
      <td>0.127557</td>
    </tr>
    <tr>
      <th>194</th>
      <td>quantity_dry</td>
      <td>0.070492</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gps_height</td>
      <td>0.061161</td>
    </tr>
    <tr>
      <th>4</th>
      <td>population</td>
      <td>0.041358</td>
    </tr>
    <tr>
      <th>195</th>
      <td>quantity_enough</td>
      <td>0.025841</td>
    </tr>
    <tr>
      <th>225</th>
      <td>waterpoint_type_other</td>
      <td>0.024601</td>
    </tr>
    <tr>
      <th>142</th>
      <td>extraction_type_group_other</td>
      <td>0.024012</td>
    </tr>
    <tr>
      <th>0</th>
      <td>amount_tsh</td>
      <td>0.018867</td>
    </tr>
    <tr>
      <th>196</th>
      <td>quantity_insufficient</td>
      <td>0.013362</td>
    </tr>
    <tr>
      <th>220</th>
      <td>waterpoint_type_communal standpipe</td>
      <td>0.011357</td>
    </tr>
    <tr>
      <th>10</th>
      <td>funder_Other</td>
      <td>0.009137</td>
    </tr>
    <tr>
      <th>223</th>
      <td>waterpoint_type_hand pump</td>
      <td>0.009076</td>
    </tr>
    <tr>
      <th>175</th>
      <td>payment_type_never pay</td>
      <td>0.008261</td>
    </tr>
    <tr>
      <th>24</th>
      <td>installer_Other</td>
      <td>0.007916</td>
    </tr>
    <tr>
      <th>221</th>
      <td>waterpoint_type_communal standpipe multiple</td>
      <td>0.007782</td>
    </tr>
    <tr>
      <th>7</th>
      <td>funder_Government Of Tanzania</td>
      <td>0.007587</td>
    </tr>
    <tr>
      <th>156</th>
      <td>management_vwc</td>
      <td>0.007440</td>
    </tr>
    <tr>
      <th>166</th>
      <td>payment_never pay</td>
      <td>0.007315</td>
    </tr>
    <tr>
      <th>20</th>
      <td>installer_DWE</td>
      <td>0.007209</td>
    </tr>
    <tr>
      <th>137</th>
      <td>extraction_type_group_gravity</td>
      <td>0.007152</td>
    </tr>
    <tr>
      <th>80</th>
      <td>permit_True</td>
      <td>0.006863</td>
    </tr>
    <tr>
      <th>141</th>
      <td>extraction_type_group_nira/tanira</td>
      <td>0.006635</td>
    </tr>
    <tr>
      <th>79</th>
      <td>permit_False</td>
      <td>0.006505</td>
    </tr>
    <tr>
      <th>197</th>
      <td>quantity_seasonal</td>
      <td>0.006108</td>
    </tr>
    <tr>
      <th>60</th>
      <td>district_code_3</td>
      <td>0.005787</td>
    </tr>
    <tr>
      <th>78</th>
      <td>public_meeting_True</td>
      <td>0.005647</td>
    </tr>
    <tr>
      <th>58</th>
      <td>district_code_1</td>
      <td>0.005548</td>
    </tr>
    <tr>
      <th>215</th>
      <td>source_type_spring</td>
      <td>0.005241</td>
    </tr>
    <tr>
      <th>59</th>
      <td>district_code_2</td>
      <td>0.005181</td>
    </tr>
    <tr>
      <th>61</th>
      <td>district_code_4</td>
      <td>0.004441</td>
    </tr>
    <tr>
      <th>77</th>
      <td>public_meeting_False</td>
      <td>0.004414</td>
    </tr>
    <tr>
      <th>27</th>
      <td>basin_Internal</td>
      <td>0.004393</td>
    </tr>
    <tr>
      <th>132</th>
      <td>construction_year_2010</td>
      <td>0.004374</td>
    </tr>
    <tr>
      <th>187</th>
      <td>water_quality_unknown</td>
      <td>0.004373</td>
    </tr>
    <tr>
      <th>81</th>
      <td>construction_year_0</td>
      <td>0.004346</td>
    </tr>
    <tr>
      <th>146</th>
      <td>extraction_type_group_submersible</td>
      <td>0.004253</td>
    </tr>
    <tr>
      <th>39</th>
      <td>region_Iringa</td>
      <td>0.004228</td>
    </tr>
    <tr>
      <th>190</th>
      <td>quality_group_good</td>
      <td>0.004167</td>
    </tr>
    <tr>
      <th>186</th>
      <td>water_quality_soft</td>
      <td>0.003968</td>
    </tr>
    <tr>
      <th>207</th>
      <td>source_spring</td>
      <td>0.003929</td>
    </tr>
    <tr>
      <th>165</th>
      <td>management_group_user-group</td>
      <td>0.003833</td>
    </tr>
    <tr>
      <th>172</th>
      <td>payment_unknown</td>
      <td>0.003716</td>
    </tr>
    <tr>
      <th>202</th>
      <td>source_machine dbh</td>
      <td>0.003714</td>
    </tr>
    <tr>
      <th>209</th>
      <td>source_type_borehole</td>
      <td>0.003690</td>
    </tr>
    <tr>
      <th>174</th>
      <td>payment_type_monthly</td>
      <td>0.003612</td>
    </tr>
    <tr>
      <th>170</th>
      <td>payment_pay per bucket</td>
      <td>0.003500</td>
    </tr>
    <tr>
      <th>179</th>
      <td>payment_type_unknown</td>
      <td>0.003479</td>
    </tr>
    <tr>
      <th>131</th>
      <td>construction_year_2009</td>
      <td>0.003443</td>
    </tr>
    <tr>
      <th>130</th>
      <td>construction_year_2008</td>
      <td>0.003428</td>
    </tr>
  </tbody>
</table>
</div>


