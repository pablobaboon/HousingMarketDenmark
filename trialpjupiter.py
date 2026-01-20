# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from pathlib import Path
import sqlite3
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import chi2_contingency
# %%
cnn=sqlite3.connect('ProjectsCPH.db')
cursor = cnn.cursor()
%load_ext sql
%sql sqlite:///ProjectsCPH.db
# %%
path = Path('C:/Users/pleon/OneDrive/Documents/Courses/Projects/Datasets/HousingCopenhagen/DKHousingPricesSample100k.csv')
print(f"Full path exists: {path.exists()}");
print(f"File exists: {path.is_file()}");
#The path exists. We can then proceed to load the data.
rawdata=pd.read_csv(path);
rawdata.to_sql('Copenhagen_Housing', cnn, if_exists='replace', index=False)
# %%
%%sql 
SELECT * FROM Copenhagen_Housing LIMIT 5 
# %%
cphallhouses=pd.read_sql_query("SELECT house_type,year_build,purchase_price,no_rooms,sqm,date FROM Copenhagen_Housing WHERE area ='Capital, Copenhagen'", cnn);
plt.plot(cphallhouses['sqm'], cphallhouses['purchase_price'], 'o')
# %%
plt.plot(cphallhouses['no_rooms'], cphallhouses['purchase_price'], 'o')
# %%
ind = cphallhouses[cphallhouses['purchase_price'].isnull()].index.tolist()
#As the result is an empty list, there are no missing values

# %%
### Lets work a bit on the statistics of the dataset, focusing mainly on the purchase price regarding the number of rooms in a same year
# %%
cph_1room = pd.read_sql_query(
	"SELECT date, no_rooms, purchase_price, sqm  "
	"FROM Copenhagen_Housing "
	"WHERE area = 'Capital, Copenhagen' AND no_rooms = 1",
	cnn,
)
cph_1room['Year_sold']=pd.to_datetime(cph_1room['date'], infer_datetime_format=True).dt.year
cph_1room.to_sql('cph_1room', cnn, if_exists='replace', index=False)
pd.read_sql_query("SELECT COUNT(Year_sold) AS Count_Sales, Year_sold FROM cph_1room GROUP BY Year_sold", cnn)
#Only in recent years, we have enough data to make relevant statistics. We can try to obtain numerically the numerical relation between square meters and price, and then between price and time
plt.plot(pd.read_sql_query("SELECT sqm FROM cph_1room WHERE Year_sold=2024",cnn),pd.read_sql_query("SELECT purchase_price FROM cph_1room WHERE Year_sold=2024",cnn),'o')
# %%
### There is already pretty reasonable stuff to analyse. The first, it is curious to see it starts in 1992, and with a high price. We can also check the
pd.read_sql_query("SELECT sqm,purchase_price, city FROM Copenhagen_Housing WHERE date LIKE '2023%' and no_rooms=1 and area='Capital, Copenhagen'", cnn)
# %%
#Lets do the agrupation. We are going to start with houses in Copenhagen City, and analyse it w.r.t ev else
kct=pd.read_sql_query("SELECT sqm,purchase_price, city,date " \
"FROM Copenhagen_Housing " \
"WHERE date LIKE '2015%' and no_rooms=1 and (city LIKE 'København%')", cnn)
print(kct)
plt.plot(kct['sqm'],kct['purchase_price']/kct['sqm'],'o')
#We understand that the variable (in 1-room apartments) we want to analyse is price/sqm, so that we isolate this dependency and we just work with the area dependency, which is fair and can be dealt with
# %%
#To automatize the process:
kct=pd.read_sql_query("SELECT sqm,purchase_price, city,date " \
"FROM Copenhagen_Housing " \
"WHERE no_rooms=1 and (city LIKE 'København%')", cnn)
kct['Year_sold']=pd.to_datetime(kct['date'], infer_datetime_format=True).dt.year
inde=[]
for i in range(len(kct['Year_sold'])):
    kct.at[i,'Price_per_sqm']=kct.at[i,'purchase_price']/kct.at[i,'sqm']
    if i!=len(kct['Year_sold'])-1:
        if kct.at[i,'Year_sold']!=kct.at[i+1,'Year_sold']:
            inde.append(i)
inde.append(len(kct['Year_sold'])-1)
#With this, we can reconstruct the different years. Now we compute the statistical parameters. 
# %%

# %%
mkct=kct.groupby('Year_sold')['Price_per_sqm'].agg(['mean', 'std', 'count']).reset_index()
plt.plot(mkct['Year_sold'], mkct['mean'], 'o', c='salmon', label='Mean Price per sqm')
plt.xlabel('Year')
plt.ylabel('Price per sqm (DKK)')
plt.legend()
plt.show()
plt.close()
plt.plot(mkct['Year_sold'], mkct['std'], 'o', c='salmon', label='Desviation per sqm')
plt.xlabel('Year')
plt.ylabel('Desviation(DKK)')
plt.legend()
plt.show()
plt.close()
plt.errorbar(mkct['Year_sold'], mkct['mean'], yerr=mkct['std'], fmt='.', ecolor='lightgray', elinewidth=3, capsize=0)
slope1, c1, r_value1, p_value1, std_err1 = stats.linregress(mkct['Year_sold'], mkct['mean'])
print(slope1, c1, r_value1, p_value1, std_err1)
priceY=slope1*mkct['Year_sold']+c1
plt.plot(mkct['Year_sold'], priceY, 'r-', label='Fitted line')
# %%
mkct
# %%
fct=pd.read_sql_query("SELECT sqm,purchase_price, city,date " \
"FROM Copenhagen_Housing " \
"WHERE no_rooms=1 and (city LIKE 'Frederiksberg C')", cnn)
fct['Year_sold']=pd.to_datetime(fct['date'], infer_datetime_format=True).dt.year
for i in range(len(fct['Year_sold'])):
    fct.at[i,'Price_per_sqm']=fct.at[i,'purchase_price']/fct.at[i,'sqm']
mfct=fct.groupby('Year_sold')['Price_per_sqm'].agg(['mean', 'std', 'count']).reset_index()
plt.plot(mfct['Year_sold'], mfct['mean'], 'o', c='salmon', label='Mean Price per sqm')
plt.xlabel('Year')
plt.ylabel('Price per sqm (DKK)')
plt.legend()
# The point in 2024 has a lot of desviation, so it's not very reliable. We should descard it for the linear regression.
mfct2 = mfct[mfct['Year_sold'] != 2024]

slope2, c2, r_value2, p_value2, std_err2 = stats.linregress(mfct2['Year_sold'], mfct2['mean'])
priceY=slope2*mfct2['Year_sold']+c2
plt.plot(mfct2['Year_sold'], priceY, 'r-', label='Fitted line')
plt.show()
plt.close()
# %%
fct2=pd.read_sql_query("SELECT sqm,purchase_price, city,date " \
"FROM Copenhagen_Housing " \
"WHERE no_rooms=1 and (city LIKE 'Frederiksberg')", cnn)
fct2['Year_sold']=pd.to_datetime(fct2['date'], infer_datetime_format=True).dt.year
for i in range(len(fct2['Year_sold'])):
    fct2.at[i,'Price_per_sqm']=fct2.at[i,'purchase_price']/fct2.at[i,'sqm']
mfct2=fct2.groupby('Year_sold')['Price_per_sqm'].agg(['mean', 'std', 'count']).reset_index()
#plt.plot(mfct2['Year_sold'], mfct2['mean'], 'o', c='salmon', label='Mean Price per sqm')
plt.errorbar(mfct2['Year_sold'], mfct2['mean'], yerr=mfct2['std'], fmt='.', ecolor='lightgray', elinewidth=3, capsize=0)
plt.xlabel('Year')
plt.ylabel('Price per sqm (DKK)')
plt.legend()
slope3, c3, r_value3, p_value3, std_err3 = stats.linregress(mfct2['Year_sold'], mfct2['mean'])
priceY=slope3*mfct2['Year_sold']+c3
plt.plot(mfct2['Year_sold'], priceY, 'r-', label='Fitted line')
plt.show()
plt.close()
# %%
print(slope1, c1, r_value1, p_value1, std_err1)
print(slope2, c2, r_value2, p_value2, std_err2)
# %%
#We now work for the closest areas from Central Copenhagen: Valby, Vanlose, Hvidovre, Rødovre
pd.read_sql_query("SELECT sqm,purchase_price, city FROM Copenhagen_Housing WHERE date LIKE '2024%' and no_rooms=1 and area='Capital, Copenhagen'", cnn)
# %%
fct3=pd.read_sql_query("SELECT sqm,purchase_price, city,date " \
"FROM Copenhagen_Housing " \
"WHERE no_rooms=1 and (city LIKE 'Valby')", cnn)
fct3['Year_sold']=pd.to_datetime(fct3['date'], infer_datetime_format=True).dt.year
for i in range(len(fct3['Year_sold'])):
    fct3.at[i,'Price_per_sqm']=fct3.at[i,'purchase_price']/fct3.at[i,'sqm']
mfct3=fct3.groupby('Year_sold')['Price_per_sqm'].agg(['mean', 'std', 'count']).reset_index()
#plt.plot(mfct2['Year_sold'], mfct2['mean'], 'o', c='salmon', label='Mean Price per sqm')
plt.errorbar(mfct3['Year_sold'], mfct3['mean'], yerr=mfct3['std'], fmt='.', ecolor='lightgray', elinewidth=1, capsize=0)
plt.xlabel('Year')
plt.ylabel('Price per sqm (DKK)')
plt.legend()
slope3, c3, r_value3, p_value3, std_err3 = stats.linregress(mfct2['Year_sold'], mfct2['mean'])
priceY=slope3*mfct2['Year_sold']+c3
plt.plot(mfct2['Year_sold'], priceY, 'r-', label='Fitted line')
plt.show()
plt.close()

fct4=pd.read_sql_query("SELECT sqm,purchase_price, city,date " \
"FROM Copenhagen_Housing " \
"WHERE no_rooms=1 and (city LIKE 'Vanløse')", cnn)
fct4['Year_sold']=pd.to_datetime(fct4['date'], infer_datetime_format=True).dt.year
for i in range(len(fct4['Year_sold'])):
    fct4.at[i,'Price_per_sqm']=fct4.at[i,'purchase_price']/fct4.at[i,'sqm']
mfct4=fct4.groupby('Year_sold')['Price_per_sqm'].agg(['mean', 'std', 'count']).reset_index()
#plt.plot(mfct2['Year_sold'], mfct2['mean'], 'o', c='salmon', label='Mean Price per sqm')
plt.errorbar(mfct4['Year_sold'], mfct4['mean'], yerr=mfct4['std'], fmt='.', ecolor='lightgray', elinewidth=1, capsize=0)
plt.xlabel('Year')
plt.ylabel('Price per sqm (DKK)')
plt.legend()
slope3, c3, r_value3, p_value3, std_err3 = stats.linregress(mfct2['Year_sold'], mfct2['mean'])
priceY=slope3*mfct2['Year_sold']+c3
plt.plot(mfct2['Year_sold'], priceY, 'r-', label='Fitted line')
plt.show()
plt.close()

fct5=pd.read_sql_query("SELECT sqm,purchase_price, city,date " \
"FROM Copenhagen_Housing " \
"WHERE no_rooms=1 and (city LIKE 'Hvidovre')", cnn)
fct5['Year_sold']=pd.to_datetime(fct5['date'], infer_datetime_format=True).dt.year
for i in range(len(fct5['Year_sold'])):
    fct5.at[i,'Price_per_sqm']=fct5.at[i,'purchase_price']/fct5.at[i,'sqm']
mfct5=fct5.groupby('Year_sold')['Price_per_sqm'].agg(['mean', 'std', 'count']).reset_index()
#plt.plot(mfct2['Year_sold'], mfct2['mean'], 'o', c='salmon', label='Mean Price per sqm')
plt.errorbar(mfct5['Year_sold'], mfct5['mean'], yerr=mfct5['std'], fmt='.', ecolor='lightgray', elinewidth=1, capsize=0)
plt.xlabel('Year')
plt.ylabel('Price per sqm (DKK)')
plt.legend()
slope3, c3, r_value3, p_value3, std_err3 = stats.linregress(mfct2['Year_sold'], mfct2['mean'])
priceY=slope3*mfct2['Year_sold']+c3
plt.plot(mfct2['Year_sold'], priceY, 'r-', label='Fitted line')
plt.show()
plt.close()

fct6=pd.read_sql_query("SELECT sqm,purchase_price, city,date " \
"FROM Copenhagen_Housing " \
"WHERE no_rooms=1 and (city LIKE 'Rødovre')", cnn)
fct6['Year_sold']=pd.to_datetime(fct6['date'], infer_datetime_format=True).dt.year
for i in range(len(fct6['Year_sold'])):
    fct6.at[i,'Price_per_sqm']=fct6.at[i,'purchase_price']/fct6.at[i,'sqm']
mfct6=fct6.groupby('Year_sold')['Price_per_sqm'].agg(['mean', 'std', 'count']).reset_index()
#plt.plot(mfct2['Year_sold'], mfct2['mean'], 'o', c='salmon', label='Mean Price per sqm')
plt.errorbar(mfct6['Year_sold'], mfct6['mean'], yerr=mfct6['std'], fmt='.', ecolor='lightgray', elinewidth=1, capsize=0)
plt.xlabel('Year')
plt.ylabel('Price per sqm (DKK)')
plt.legend()
slope3, c3, r_value3, p_value3, std_err3 = stats.linregress(mfct2['Year_sold'], mfct2['mean'])
priceY=slope3*mfct2['Year_sold']+c3
plt.plot(mfct2['Year_sold'], priceY, 'r-', label='Fitted line')
plt.show()
plt.close()
# %%
#To end up the preliminary analysis of the 1-room apartments in Copenhagen, we plot all this areas in the same graph

# %%
plt.errorbar(mkct['Year_sold'], mkct['mean'], yerr=mkct['std'], fmt='.r', ecolor='r', elinewidth=0, capsize=0)
plt.errorbar(mfct['Year_sold'], mfct['mean'], yerr=mfct['std'], fmt='.b', ecolor='b', elinewidth=0, capsize=0)
plt.errorbar(mfct2['Year_sold'], mfct2['mean'], yerr=mfct2['std'], fmt='.g', ecolor='g', elinewidth=0, capsize=0)
plt.errorbar(mfct3['Year_sold'], mfct3['mean'], yerr=mfct3['std'], fmt='.y', ecolor='y', elinewidth=0, capsize=0)
plt.errorbar(mfct4['Year_sold'], mfct4['mean'], yerr=mfct4['std'], fmt='.m', ecolor='pink', elinewidth=0, capsize=0)
plt.errorbar(mfct5['Year_sold'], mfct5['mean'], yerr=mfct5['std'], fmt='.c', ecolor='salmon', elinewidth=0, capsize=0)
plt.errorbar(mfct6['Year_sold'], mfct6['mean'], yerr=mfct6['std'], fmt='.k', ecolor='black', elinewidth=0, capsize=0)
plt.xlabel('Year')
plt.ylabel('Price per sqm (DKK)')
# %%
#Before we continue to 2-room appartments, we write a distribution over the years for the number of rooms and the number of sales in Copenhagen Region
plt.bar(cph_rooms1992['no_rooms'], cph_rooms1992['Number_of_Sales']/cph_rooms1992['Number_of_Sales'].sum(), color='salmon')
# %%
fig, ax = plt.subplots(6,6, figsize=(15,15))
cph_rooms1992=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '1992%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[0,0].bar(cph_rooms1992['no_rooms'], cph_rooms1992['Number_of_Sales']/cph_rooms1992['Number_of_Sales'].sum(), color='salmon')
cph_rooms1993=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '1993%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[0,1].bar(cph_rooms1993['no_rooms'], cph_rooms1993['Number_of_Sales']/cph_rooms1993['Number_of_Sales'].sum(), color='salmon')
cph_rooms1994=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '1994%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[0,2].bar(cph_rooms1994['no_rooms'], cph_rooms1994['Number_of_Sales']/cph_rooms1994['Number_of_Sales'].sum(), color='salmon')
cph_rooms1995=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '1995%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[0,3].bar(cph_rooms1995['no_rooms'], cph_rooms1995['Number_of_Sales']/cph_rooms1995['Number_of_Sales'].sum(), color='salmon')
cph_rooms1996=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '1996%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[0,4].bar(cph_rooms1996['no_rooms'], cph_rooms1996['Number_of_Sales']/cph_rooms1996['Number_of_Sales'].sum(), color='salmon')
cph_rooms1997=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '1997%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[0,5].bar(cph_rooms1997['no_rooms'], cph_rooms1997['Number_of_Sales']/cph_rooms1997['Number_of_Sales'].sum(), color='salmon')
cph_rooms1998=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '1998%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[1,0].bar(cph_rooms1998['no_rooms'], cph_rooms1998['Number_of_Sales']/cph_rooms1998['Number_of_Sales'].sum(), color='salmon')
cph_rooms1999=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '1999%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[1,1].bar(cph_rooms1999['no_rooms'], cph_rooms1999['Number_of_Sales']/cph_rooms1999['Number_of_Sales'].sum(), color='salmon')
cph_rooms2000=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2000%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[1,2].bar(cph_rooms2000['no_rooms'], cph_rooms2000['Number_of_Sales']/cph_rooms2000['Number_of_Sales'].sum(), color='salmon')
cph_rooms2001=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2001%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[1,3].bar(cph_rooms2001['no_rooms'], cph_rooms2001['Number_of_Sales']/cph_rooms2001['Number_of_Sales'].sum(), color='salmon')
cph_rooms2002=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2002%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[1,4].bar(cph_rooms2002['no_rooms'], cph_rooms2002['Number_of_Sales']/cph_rooms2002['Number_of_Sales'].sum(), color='salmon')
cph_rooms2003=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2003%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[1,4].bar(cph_rooms2003['no_rooms'], cph_rooms2003['Number_of_Sales']/cph_rooms2003['Number_of_Sales'].sum(), color='salmon')
cph_rooms2004=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2004%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[1,5].bar(cph_rooms2004['no_rooms'], cph_rooms2004['Number_of_Sales']/cph_rooms2004['Number_of_Sales'].sum(), color='salmon')
cph_rooms2005=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2005%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[2,0].bar(cph_rooms2005['no_rooms'], cph_rooms2005['Number_of_Sales']/cph_rooms2005['Number_of_Sales'].sum(), color='salmon')
cph_rooms2006=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2006%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[2,1].bar(cph_rooms2006['no_rooms'], cph_rooms2006['Number_of_Sales']/cph_rooms2006['Number_of_Sales'].sum(), color='salmon')
cph_rooms2007=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2007%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[2,2].bar(cph_rooms2007['no_rooms'], cph_rooms2007['Number_of_Sales']/cph_rooms2007['Number_of_Sales'].sum(), color='salmon')
cph_rooms2008=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2008%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[2,3].bar(cph_rooms2008['no_rooms'], cph_rooms2008['Number_of_Sales']/cph_rooms2008['Number_of_Sales'].sum(), color='salmon')
cph_rooms2009=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2009%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[2,4].bar(cph_rooms2009['no_rooms'], cph_rooms2009['Number_of_Sales']/cph_rooms2009['Number_of_Sales'].sum(), color='salmon')
cph_rooms2010=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2010%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[2,5].bar(cph_rooms2010['no_rooms'], cph_rooms2010['Number_of_Sales']/cph_rooms2010['Number_of_Sales'].sum(), color='salmon')
cph_rooms2011=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2011%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[3,0].bar(cph_rooms2011['no_rooms'], cph_rooms2011['Number_of_Sales']/cph_rooms2011['Number_of_Sales'].sum(), color='salmon')
cph_rooms2012=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2012%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[3,1].bar(cph_rooms2012['no_rooms'], cph_rooms2012['Number_of_Sales']/cph_rooms2012['Number_of_Sales'].sum(), color='salmon')
cph_rooms2013=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2013%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[3,2].bar(cph_rooms2013['no_rooms'], cph_rooms2013['Number_of_Sales']/cph_rooms2013['Number_of_Sales'].sum(), color='salmon')
cph_rooms2014=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2014%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[3,3].bar(cph_rooms2014['no_rooms'], cph_rooms2014['Number_of_Sales']/cph_rooms2014['Number_of_Sales'].sum(), color='salmon')
cph_rooms2015=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2015%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[3,4].bar(cph_rooms2015['no_rooms'], cph_rooms2015['Number_of_Sales']/cph_rooms2015['Number_of_Sales'].sum(), color='salmon')
cph_rooms2016=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2016%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[3,5].bar(cph_rooms2016['no_rooms'], cph_rooms2016['Number_of_Sales']/cph_rooms2016['Number_of_Sales'].sum(), color='salmon')
cph_rooms2017=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2017%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[4,0].bar(cph_rooms2017['no_rooms'], cph_rooms2017['Number_of_Sales']/cph_rooms2017['Number_of_Sales'].sum(), color='salmon')
cph_rooms2018=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2018%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[4,1].bar(cph_rooms2018['no_rooms'], cph_rooms2018['Number_of_Sales']/cph_rooms2018['Number_of_Sales'].sum(), color='salmon')
cph_rooms2019=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2019%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[4,2].bar(cph_rooms2019['no_rooms'], cph_rooms2019['Number_of_Sales']/cph_rooms2019['Number_of_Sales'].sum(), color='salmon')
cph_rooms2020=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2020%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[4,3].bar(cph_rooms2020['no_rooms'], cph_rooms2020['Number_of_Sales']/cph_rooms2020['Number_of_Sales'].sum(), color='salmon')
cph_rooms2021=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2021%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[4,4].bar(cph_rooms2021['no_rooms'], cph_rooms2021['Number_of_Sales']/cph_rooms2021['Number_of_Sales'].sum(), color='salmon')
cph_rooms2022=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2022%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[4,5].bar(cph_rooms2022['no_rooms'], cph_rooms2022['Number_of_Sales']/cph_rooms2022['Number_of_Sales'].sum(), color='salmon')
cph_rooms2023=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2023%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[5,2].bar(cph_rooms2023['no_rooms'], cph_rooms2023['Number_of_Sales']/cph_rooms2023['Number_of_Sales'].sum(), color='salmon')
cph_rooms2024=pd.read_sql_query("SELECT no_rooms, COUNT(no_rooms) AS Number_of_Sales FROM Copenhagen_Housing WHERE area='Capital, Copenhagen' AND date LIKE '2024%' AND house_type!='Summerhouse' GROUP BY no_rooms", cnn)
ax[5,3].bar(cph_rooms2024['no_rooms'], cph_rooms2024['Number_of_Sales']/cph_rooms2024['Number_of_Sales'].sum(), color='salmon')

#The assymetry is clear and peaks at 3-4 rooms appartments. To see the evolution during the years, we can do a copilation of different years. Let's do all years from 1992 to 2024
# %%
#Interesting enough, we can track the relative frequency of the 1,2,3,4,5 and 6 rooms appartments during the years in Copenhagen Region.
#We also take out the summerhus, as they are not residential houses perse. We build up the dataframe.
years=list(range(1992,2025))
norooms=list(range(1,7))
relfreqs=[]
listhdf=[cph_rooms1992, cph_rooms1993, cph_rooms1994, cph_rooms1995, cph_rooms1996, cph_rooms1997, cph_rooms1998, cph_rooms1999, cph_rooms2000,
         cph_rooms2001, cph_rooms2002, cph_rooms2003, cph_rooms2004, cph_rooms2005, cph_rooms2006, cph_rooms2007, cph_rooms2008,
         cph_rooms2009, cph_rooms2010, cph_rooms2011, cph_rooms2012, cph_rooms2013, cph_rooms2014, cph_rooms2015, cph_rooms2016, cph_rooms2017,
         cph_rooms2018, cph_rooms2019, cph_rooms2020, cph_rooms2021, cph_rooms2022, cph_rooms2023, cph_rooms2024]
for i in range(len(listhdf)):
    relfreqs.append(listhdf[i]['Number_of_Sales']/listhdf[i]['Number_of_Sales'].sum())
    relfreqs[i] = relfreqs[i].drop(relfreqs[i][relfreqs[i].index > 5].index)
relfreqsdf=pd.DataFrame(relfreqs, index=years)
relfreqsdf.columns=norooms
print(relfreqsdf)
relfreqsdf.plot(kind='line', marker='o')
#We funily enough see the increase in the ammount of 2 and 3-room appartments sold during the past year, while 4,5 and 6 have decreased. Still, most of the soles are 2,3 and 4. This is probably becuase of the money. 
# %%
### We have approximated the problem. Now it's time to start with the real statistical analysis of the problem. The first step is to see the distribution of prices for each year and each room, and plot them all together to see the time evolution on the distribution of prices. For that, we build up an interval table
# %%

#cphpr1_1992=pd.read_sql_query("SELECT sqm,purchase_price FROM Copenhagen_Housing WHERE no_rooms=1 and city LIKE 'København%' and date LIKE '2024%'", cnn)
#for i in range(len(cphpr1_1992['sqm'])):
#    cphpr1_1992.at[i,'Price_per_sqm']=cphpr1_1992.at[i,'purchase_price']/cphpr1_1992.at[i,'sqm']
#print(cphpr1_1992)
#Which years have enough information so that we can actually build a histogram? Lets first see this
rawdata['Year_sold']=pd.to_datetime(rawdata['date'], infer_datetime_format=True).dt.year
rawdata.to_sql('Copenhagen_Housing', cnn, if_exists='replace', index=False)
frequency_rooms=pd.read_sql_query("SELECT Year_sold, COUNT(purchase_price) AS Number_of_Sales FROM Copenhagen_Housing WHERE city LIKE 'København%' AND house_type!='Summerhouse' AND no_rooms=1 GROUP BY Year_sold", cnn)
years_df=pd.DataFrame({'Year_sold':years})
frequency_rooms_complete = years_df.merge(frequency_rooms, on='Year_sold', how='left')
frequency_rooms_complete['Number_of_Sales'] = frequency_rooms_complete['Number_of_Sales'].fillna(0).astype(int)
#From 2019 to 2025, we can do relevant statistics. Automatizing now the year vectors in which we have enough data
stl=[]
years_enough_data = frequency_rooms_complete[frequency_rooms_complete['Number_of_Sales'] >= 15]['Year_sold'].tolist()
for i in years_enough_data:
    stl.append(pd.read_sql_query("SELECT purchase_price/sqm AS Price_per_sqm FROM Copenhagen_Housing WHERE no_rooms=1 and city LIKE 'København%' and Year_sold="+str(i), cnn))
#We study the range of the variable (prize/sqm) to build the intervals
print("2019:",stl[0].min(),stl[0].max())
print("2021:",stl[1].min(),stl[1].max())
print("2022:",stl[2].min(),stl[2].max())
#Range from 4250 to 74450. Now, buil intervals. 
np.sqrt(len(stl[0]))#~around 5 intervals
np.sqrt(len(stl[1]))#~around 5 intervals
np.sqrt(len(stl[2]))#~around 5 intervals
intervals = np.linspace(4000, 75000, num=5)
for i in range(len(stl[0])):
    stl[0].at[i,'Interval']=pd.cut([stl[0].at[i,'Price_per_sqm']], bins=intervals, labels=False)[0]
for i in range(len(stl[1])):
    stl[1].at[i,'Interval']=pd.cut([stl[1].at[i,'Price_per_sqm']], bins=intervals, labels=False)[0]
for i in range(len(stl[2])):
    stl[2].at[i,'Interval']=pd.cut([stl[2].at[i,'Price_per_sqm']], bins=intervals, labels=False)[0]
freq2019=stl[0].groupby('Interval')['Price_per_sqm'].agg(['count']).reset_index()
freq2021=stl[1].groupby('Interval')['Price_per_sqm'].agg(['count']).reset_index()
freq2022=stl[2].groupby('Interval')['Price_per_sqm'].agg(['count']).reset_index()
interval_df=pd.DataFrame({'Interval_Center':[(intervals[i]+intervals[i+1])/2 for i in range(len(intervals)-1)]})
interval_df['Interval'] = interval_df.index
freq2019=interval_df.merge(freq2019, on='Interval', how='left').fillna(0)
freq2021=interval_df.merge(freq2021, on='Interval', how='left').fillna(0)
freq2022=interval_df.merge(freq2022, on='Interval', how='left').fillna(0)
#ax = plt.hist(freq2019['count'], bins=5, color='salmon', alpha=0.7,density=True)
ax = freq2019.plot(x='Interval_Center', y='count', kind='bar', 
                   color='salmon', alpha=0.7, legend=False)
plt.title('Price Distribution - 2021 (Bar Plot)')
plt.xlabel('Price Intervals')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

ax = freq2021.plot(x='Interval_Center', y='count', kind='bar', 
                   color='salmon', alpha=0.7, legend=False)
plt.title('Price Distribution - 2021 (Bar Plot)')
plt.xlabel('Price Intervals')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()
ax = freq2022.plot(x='Interval_Center', y='count', kind='bar', 
                   color='salmon', alpha=0.7, legend=False)
plt.title('Price Distribution - 2022 (Bar Plot)')
plt.xlabel('Price Intervals')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()
# Evolution of the price distribution during the years. Clear shift to the right, as expected. We can also analyse it as a temporal funtion of the momenta. With respect to centrale values, we see that mean are median should give more or less similar results. 
# %%
## We can start with the 2-room appartments now
# %%
kbhc2r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold FROM Copenhagen_Housing WHERE no_rooms=2 and city LIKE 'København%'", cnn)
kbhc2r_desc=kbhc2r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
#print(kbhc2r_desc)#WTF happens for year 1997? lol
kbhc2r[kbhc2r['Year_sold']==1997]#Two very high values that skew the mean a lot. It is most interesting, in our case, to use the median instead of the mean. For large data, both descriptions are good enough. 
kbhc2r[kbhc2r['Year_sold']==1993]#In this case, actually the mean is more reliable, as there is a pretty uneven distribution of cases. We can graphically plot this like in the following. 
#To graphically see this, we can plot the distribution of prices plus menan plus median for each year
for k in years:
    plt.hist(kbhc2r[kbhc2r['Year_sold']==k]['purchase_price/sqm'], bins=int(np.sqrt(len(kbhc2r[kbhc2r['Year_sold']==1997]['purchase_price/sqm'])).round()+1), color='salmon', alpha=0.7,histtype='step')
    plt.show()
    plt.close()
#From here we learn visualise some stuff about the dispersion, not being a good statistical indicative. There is also the problem, the dispersion is not a good meassure for early years due to this. Discussed this, we now plot 
# %%
vecmeds=kbhc2r_desc['mean'];
vecmeds[0:9]=kbhc2r_desc['median'][0:9]
plt.errorbar(kbhc2r_desc['Year_sold'],vecmeds,yerr=kbhc2r_desc['std'],fmt='.', ecolor='lightgray', elinewidth=1, capsize=0)
# %%
#Repeat process for Valby, Rødovre, Hvidovre, Frederiskberg C and Frederiksberg!
fredc2r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=2 and city LIKE 'Frederiksberg C'", cnn)
fredc2r_desc=fredc2r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
#print(fredc2r_desc)
#print(fredc2r)
#Two years to take care. 2018 and 2019 I will also authomatize this process for this ato some point. 
fredc2r[fredc2r['Year_sold']==2018] #Mean captures better the distribution 
fredc2r[fredc2r['Year_sold']==2019] #Mean also captures more or less better the distribution.
plt.plot(fredc2r_desc['Year_sold'],fredc2r_desc['mean'],'o')
# %%
fred2r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=2 and city LIKE 'Frederiksberg'", cnn)
fred2r_desc=fred2r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
#print(fred2r_desc)#2002 we use the median.
vecmeds2=fred2r_desc['mean']
vecmeds2[fred2r_desc['Year_sold']==2002]=fred2r_desc['median'][fred2r_desc['Year_sold']==2002]
plt.plot(fred2r_desc['Year_sold'],vecmeds2,'o')

# %%
valb2r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=2 and city LIKE 'Valby'", cnn)
valb2r_desc=valb2r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
print(valb2r_desc)
plt.plot(valb2r_desc['Year_sold'],valb2r_desc['mean'],'o')
#From 2013 onwards, statistics make sense. Before that, we can clearly see there is hardly enough data to arrive into any conclusion
# %%
Hvid2r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=2 and city LIKE 'Hvidovre'", cnn)
Hvid2r_desc=Hvid2r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
print(Hvid2r_desc)
plt.plot(Hvid2r_desc['Year_sold'],Hvid2r_desc['mean'],'o')
# %%
Rod2r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=2 and city LIKE 'Rødovre'", cnn)
Rod2r_desc=Rod2r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
print(Rod2r_desc)
plt.plot(Rod2r_desc[Rod2r_desc['Year_sold'] != 2003]['Year_sold'], Rod2r_desc[Rod2r_desc['Year_sold'] != 2003]['mean'], 'o')

# %%
Van2r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=2 and city LIKE 'Vanløse'", cnn)
Van2r_desc=Rod2r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
print(Van2r_desc)
plt.plot(Van2r_desc[Van2r_desc['Year_sold'] != 2003]['Year_sold'], Van2r_desc[Van2r_desc['Year_sold'] != 2003]['mean'], 'o')

# %%
#Now, we can plot all of them at the same time and see the evolution
plt.plot(kbhc2r_desc['Year_sold'],vecmeds,'-o',c='darkblue')
plt.plot(fredc2r_desc['Year_sold'],fredc2r_desc['mean'],'-o',c='mediumblue')
plt.plot(fred2r_desc['Year_sold'],vecmeds2,'-o',c='blue')
plt.plot(valb2r_desc['Year_sold'],valb2r_desc['mean'],'-o',c='darkturquoise')
plt.plot(Hvid2r_desc['Year_sold'],Hvid2r_desc['mean'],'-o',c='cyan')
plt.plot(Rod2r_desc[Rod2r_desc['Year_sold'] != 2003]['Year_sold'], Rod2r_desc[Rod2r_desc['Year_sold'] != 2003]['mean'], '-o',c='dodgerblue')
plt.plot(Van2r_desc[Van2r_desc['Year_sold'] != 2003]['Year_sold'], Van2r_desc[Van2r_desc['Year_sold'] != 2003]['mean'], '-o',c='royalblue')

# %%
#We are gonna do a second thing, which is basically ploting the 1room and 2rooms in Copenhagen city if there is relation between them
plt.plot(mkct['Year_sold'], mkct['mean'], '-o', c='salmon', label='Mean Price per sqm')
plt.plot(kbhc2r_desc['Year_sold'],vecmeds,'-o',c='darkblue')
#We can consider that they can come from the same generating function, more or less. 
# %%
#Finally, we can try to see the evolution of the standart deviation.
plt.plot(kbhc2r_desc['Year_sold'][10:],kbhc2r_desc['std'][10:],'-o')
plt.show()
plt.close()
plt.plot(fred2r_desc['Year_sold'],fred2r_desc['std'],'-o')
plt.show()
plt.close()
#In Frederiksberg, there is no enough data to quantize the deviation. In any way, we see that 
# %%
def Centralizer(avg,med,x,eps):
    sigyears=[]; ind=[]
    if len(avg)==len(med)==len(x):
        for i in range(len(avg)):
            if abs(avg[i]-med[i])>eps:
                sigyears.append(x[i])
                ind.append(i)
    return(sigyears)
# %%
kbhc3r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold FROM Copenhagen_Housing WHERE no_rooms=3 and city LIKE 'København%'", cnn)
kbhc3r_desc=kbhc3r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
#print(kbhc3r_desc)
#ytc=Centralizer(kbhc3r_desc['mean'],kbhc3r_desc['median'],kbhc3r_desc['Year_sold'],5000)
#for i in range(len(ytc)):
    #print(kbhc3r['purchase_price/sqm'][kbhc3r['Year_sold']==ytc[i]],ytc[i],kbhc3r_desc['mean'][kbhc3r_desc['Year_sold']==ytc[i]],kbhc3r_desc['median'][kbhc3r_desc['Year_sold']==ytc[i]])
plt.plot(kbhc3r_desc['Year_sold'],kbhc3r_desc['mean'],'-o',c='salmon')
plt.plot(kbhc3r_desc['Year_sold'],kbhc3r_desc['median'],'-o',c='darkblue')
plt.plot(kbhc2r_desc['Year_sold'],kbhc2r_desc['mean'],'-o',c='red')
plt.plot(kbhc2r_desc['Year_sold'],kbhc2r_desc['median'],'-o',c='yellow')

# %%
#Let's say everywhere mean, for the sake of the distribution.
fredc3r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=3 and city LIKE 'Frederiksberg C'", cnn)
fredc3r_desc=fredc3r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
plt.plot(fredc3r_desc['Year_sold'],fredc3r_desc['mean'],'-o',c='salmon')
plt.plot(fredc3r_desc['Year_sold'],fredc3r_desc['median'],'-o',c='darkblue')
plt.show()
plt.close()
plt.plot(fredc3r_desc['Year_sold'],fredc3r_desc['std'],'o',c='darkblue')
# %%
fred3r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=3 and city LIKE 'Frederiksberg'", cnn)
fred3r_desc=fred3r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
plt.plot(fred3r_desc['Year_sold'],fred3r_desc['mean'],'-o',c='salmon')
plt.plot(fred3r_desc['Year_sold'],fred3r_desc['median'],'-o',c='darkblue')
plt.show()
plt.close()
plt.plot(fred3r_desc['Year_sold'],fred3r_desc['std'],'o',c='darkblue')

# %%
valb3r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=3 and city LIKE 'Valby'", cnn)
valb3r_desc=valb3r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
plt.plot(valb3r_desc['Year_sold'],valb3r_desc['mean'],'o')
# %%
Hvid3r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=3 and city LIKE 'Hvidovre'", cnn)
Hvid3r_desc=Hvid3r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
#print(Hvid3r_desc)
plt.plot(Hvid3r_desc['Year_sold'],Hvid3r_desc['mean'],'-o')
# %%
Rod3r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=3 and city LIKE 'Rødovre'", cnn)
Rod3r_desc=Rod3r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
#print(Rod3r_desc)
plt.plot(Rod3r_desc[Rod3r_desc['Year_sold'] != 2003]['Year_sold'], Rod3r_desc[Rod3r_desc['Year_sold'] != 2003]['mean'], '-o')

# %%
Van3r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold,city FROM Copenhagen_Housing WHERE no_rooms=3 and city LIKE 'Vanløse'", cnn)
Van3r_desc=Van3r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
#print(Van3r_desc)
plt.plot(Van3r_desc[Van3r_desc['Year_sold'] != 2003]['Year_sold'], Van3r_desc[Van3r_desc['Year_sold'] != 2003]['mean'], '-o')
# %%
kbhc4r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold FROM Copenhagen_Housing WHERE no_rooms=4 and city LIKE 'København%'", cnn)
kbhc4r_desc=kbhc4r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
kbhc5r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold FROM Copenhagen_Housing WHERE no_rooms=5 and city LIKE 'København%'", cnn)
kbhc5r_desc=kbhc5r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
kbhc6r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold FROM Copenhagen_Housing WHERE no_rooms=6 and city LIKE 'København%'", cnn)
kbhc6r_desc=kbhc6r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
kbhc1r=pd.read_sql_query("SELECT purchase_price/sqm, Year_sold FROM Copenhagen_Housing WHERE no_rooms=1 and city LIKE 'København%'", cnn)
kbhc1r_desc=kbhc1r.groupby('Year_sold')['purchase_price/sqm'].agg(['mean', 'std', 'count', 'median']).reset_index()
#plt.plot(kbhc6r_desc['Year_sold'],kbhc6r_desc['mean'],'-o',c='lightblue')
plt.plot(kbhc5r_desc['Year_sold'],kbhc5r_desc['mean'],'-o',c='darkblue')
plt.plot(kbhc4r_desc['Year_sold'],kbhc4r_desc['mean'],'-o',c='yellow')
#plt.plot(kbhc4r_desc['Year_sold'],kbhc4r_desc['median'],'-o',c='lightblue')
plt.plot(kbhc3r_desc['Year_sold'],kbhc3r_desc['mean'],'-o',c='salmon')
#plt.plot(kbhc3r_desc['Year_sold'],kbhc3r_desc['median'],'-o',c='darkblue')
plt.plot(kbhc2r_desc['Year_sold'],kbhc2r_desc['mean'],'-o',c='red')
#plt.plot(kbhc1r_desc['Year_sold'],kbhc1r_desc['mean'],'-o',c='brown')
#plt.plot(kbhc2r_desc['Year_sold'],kbhc2r_desc['median'],'-o',c='yellow')


#Now probably after optimization, we can analyse the increase as a power law over time. We take logarithms to see.
# %%
plt.plot(kbhc4r_desc['Year_sold'],np.log(kbhc4r_desc['mean']),'-o',c='yellow')
plt.plot(kbhc3r_desc['Year_sold'],np.log(kbhc3r_desc['mean']),'-o',c='red')
plt.plot(kbhc2r_desc['Year_sold'],np.log(kbhc2r_desc['mean']),'-o',c='blue')
plt.plot(kbhc4r_desc['Year_sold'][20:],pricer4,'-',c='yellow')
plt.plot(kbhc3r_desc['Year_sold'][20:],pricer3,'-',c='red')
plt.plot(kbhc2r_desc['Year_sold'][20:],pricer2,'-',c='blue')

#Empezar plots desde 2012 para ver la recta.

# %%
sloper4, cr4, r_valuer4, p_valuer4, std_errr4 = stats.linregress(kbhc4r_desc['Year_sold'][20:],np.log(kbhc4r_desc['mean'][20:]))
pricer4=sloper4*kbhc4r_desc['Year_sold'][20:]+cr4
sloper3, cr3, r_valuer3, p_valuer3, std_errr3 = stats.linregress(kbhc3r_desc['Year_sold'][20:],np.log(kbhc3r_desc['mean'][20:]))
pricer3=sloper3*kbhc3r_desc['Year_sold'][20:]+cr3
sloper2, cr2, r_valuer2, p_valuer2, std_errr2 = stats.linregress(kbhc2r_desc['Year_sold'][20:],np.log(kbhc2r_desc['mean'][20:]))
pricer2=sloper2*kbhc2r_desc['Year_sold'][20:]+cr2

# %%
plt.plot(kbhc4r_desc['Year_sold'][20:],pricer4,'-',c='yellow')
# %%
print(sloper4,sloper3,sloper2)
print(r_valuer4,r_valuer3,r_valuer2)
print(p_valuer4,p_valuer3,p_valuer2)
# %%
exp2r=np.exp(cr2+sloper2*kbhc2r_desc['Year_sold'][18:])
exp3r=np.exp(cr3+sloper3*kbhc3r_desc['Year_sold'][18:])
exp4r=np.exp(cr4+sloper4*kbhc4r_desc['Year_sold'][18:])
plt.plot(kbhc4r_desc['Year_sold'],kbhc4r_desc['mean'],'-o',c='green')
plt.plot(kbhc3r_desc['Year_sold'],kbhc3r_desc['mean'],'-o',c='darkblue')
plt.plot(kbhc2r_desc['Year_sold'],kbhc2r_desc['mean'],'-o',c='red')
plt.plot(kbhc2r_desc['Year_sold'][18:],exp2r,'-',c='red')
plt.plot(kbhc3r_desc['Year_sold'][18:],exp3r,'-',c='darkblue')
plt.plot(kbhc4r_desc['Year_sold'][18:],exp4r,'-',c='green')
# %%
#Things left to do with this. Hypothesis: data does not depend on the number of rooms. After that, continue a bit with the model
# Once done the time dependency, maybe it would be cool to look at the rest of the dependencies, and try to finish the model
# We can now proceed to continue with the rest of the houses in the outskirts of the city, to visualize everything
# %%
#inf=pd.read_sql_query("SELECT 'dk_ann_infl_rate%' FROM Copenhagen_Housing WHERE city LIKE 'København%'", cnn)
#print(inf)
infdata=rawdata[['dk_ann_infl_rate%','Year_sold']].groupby('Year_sold')['dk_ann_infl_rate%'].agg(['mean']).reset_index()
#print(infdata)
#plt.plot(infdata['Year_sold'],infdata['mean'],'-o')
#We compute the cumulative inflation rate easily as the prod(inf+1)-1. Basically, we have to compound them.
prodinf=infdata['mean']/100+1
cuminf=[]
i=len(infdata['mean'])-2
cuminf.append(prodinf[i])
while i>=0:
    cuminf.append(m.prod(prodinf[i:]))
    i=i-1
cuminf_df=pd.DataFrame({'Year_sold':years,'CumuInf':cuminf})
#plt.plot(cuminf_df['Year_sold'],cuminf_df['CumuInf'],'.')
#Now, with the cumulative inflation computed, we can see the evolution of the prices, if just they had been affected by inflation
#We plot the 3 and 4 room inflation lines 
plt.plot(kbhc4r_desc['Year_sold'],kbhc4r_desc['mean'],'-o',c='green')
plt.plot(kbhc3r_desc['Year_sold'],kbhc3r_desc['mean'],'-o',c='darkblue')
plt.plot(cuminf_df['Year_sold'],cuminf_df['CumuInf']*kbhc4r_desc['mean'][1]/cuminf_df['CumuInf'][1],'--',c='green')
plt.plot(cuminf_df['Year_sold'],cuminf_df['CumuInf']*kbhc3r_desc['mean'][1]/cuminf_df['CumuInf'][1],'--',c='darkblue')
#This analysis is finished, as we notice already that inflation
# %%
#path2=Path('C:/Users/pleon/OneDrive/Documents/Courses/Projects/Datasets/HousingCopenhagen/SalariesCopenhagen.csv')
#SalariesCPH = pd.read_csv(path2, sep=None, engine='python')
SalariesCPH=np.array([39739.29,40115.26,40747.93,41249.51,42350.03,43249.13,44242.81,45381.62,46257.01,47710.98,49458.15,52454.51])
yearsal=list(range(2013,2025))
SalariesCPH_norm=(SalariesCPH-SalariesCPH.min())/(SalariesCPH.max()-SalariesCPH.min())
#cuminf_norm2013=(cuminf_df['CumuInf'][21:]-cuminf_df['CumuInf'][21:].min())/(cuminf_df['CumuInf'][21:].max()-cuminf_df['CumuInf'][21:].min())
#plt.plot(yearsal,SalariesCPH_norm,'-o')
#plt.plot(cuminf_df['Year_sold'][21:],cuminf_norm2013)
SalaryValue=SalariesCPH.min()*cuminf_df['CumuInf'][21:]/cuminf_df['CumuInf'][21]
plt.plot(yearsal,SalariesCPH,'-o')
plt.plot(yearsal,SalaryValue,'-o')
# %%
#cuminf_df['Year_sold'][cuminf_df['Year_sold']==2013]
#kbhc4r_desc_a=kbhc4r_desc['mean'][21:]-kbhc4r_desc['mean'][21:].min())/(kbhc4r_desc['mean'][21:].max()-kbhc4r_desc['mean'][21:].min())
#kbhc3r_desc_a=kbhc3r_desc['mean'][21:]-kbhc3r_desc['mean'][21:].min())/(kbhc3r_desc['mean'][21:].max()-kbhc3r_desc['mean'][21:].min())
kbhc4r_desc_a=np.array(kbhc4r_desc['mean'][21:])
kbhc3r_desc_a=np.array(kbhc3r_desc['mean'][21:])


# %%

#problematic: Tipifying does not work. We have to think on another stuff. All of them seemed to have a pretty good expontential behavior, so what we can compute is the ratio of the vector derivaties with respect to x.
der4rooms=[]
der3rooms=[]
dersalaries=[]
for i in range(len(kbhc4r_desc_a)):
    if i == 0:
        # Second order forward difference: -f(x+2) + 4f(x+1) - 3f(x) all divided by 2
        der4rooms.append((-kbhc4r_desc_a[2] + 4*kbhc4r_desc_a[1] - 3*kbhc4r_desc_a[0]) / 2)
    elif i == len(kbhc4r_desc_a) - 1:
        # Second order backward difference: 3f(x) - 4f(x-1) + f(x-2) all divided by 2
        der4rooms.append((3*kbhc4r_desc_a[i] - 4*kbhc4r_desc_a[i-1] + kbhc4r_desc_a[i-2]) / 2)
    else:
        # Central difference (same as before)
        der4rooms.append((kbhc4r_desc_a[i+1] - kbhc4r_desc_a[i-1]) / 2)

for i in range(len(kbhc3r_desc_a)):
    if i == 0:
        # Second order forward difference: -f(x+2) + 4f(x+1) - 3f(x) all divided by 2
        der3rooms.append((-kbhc3r_desc_a[2] + 4*kbhc3r_desc_a[1] - 3*kbhc3r_desc_a[0]) / 2)
    elif i == len(kbhc3r_desc_a) - 1:
        # Second order backward difference: 3f(x) - 4f(x-1) + f(x-2) all divided by 2
        der3rooms.append((3*kbhc3r_desc_a[i] - 4*kbhc3r_desc_a[i-1] + kbhc3r_desc_a[i-2]) / 2)
    else:
        # Central difference (same as before)
        der3rooms.append((kbhc3r_desc_a[i+1] - kbhc3r_desc_a[i-1]) / 2)

for i in range(len(SalariesCPH)):
    if i == 0:
        # Second order forward difference: -f(x+2) + 4f(x+1) - 3f(x) all divided by 2
        dersalaries.append((-SalariesCPH[2] + 4*SalariesCPH[1] - 3*SalariesCPH[0]) / 2)
    elif i == len(SalariesCPH) - 1:
        # Second order backward difference: 3f(x) - 4f(x-1) + f(x-2) all divided by 2
        dersalaries.append((3*SalariesCPH[i] - 4*SalariesCPH[i-1] + SalariesCPH[i-2]) / 2)
    else:
        # Central difference (same as before)
        dersalaries.append((SalariesCPH[i+1] - SalariesCPH[i-1]) / 2)

plt.plot(yearsal,np.array(der3rooms)/kbhc3r_desc_a,'o-')
plt.plot(yearsal,np.array(der4rooms)/kbhc4r_desc_a,'o-')
plt.plot(yearsal,np.array(dersalaries)/SalariesCPH,'o-')


    
# %%
print(np.mean(np.array(der3rooms)/kbhc3r_desc_a))
print(np.mean(np.array(der4rooms)/kbhc4r_desc_a))
print(np.mean(np.array(dersalaries)/SalariesCPH))
print(sloper3)
print(sloper4)
#The system illustrates how the exponential increase of the houses has been much higher than than the one of the salaries. This illustrates what we wanted
# %%
#Next things to do: Hypothesis contrasts between different populatons. Have to write two things: chi^2 test for 
#knowing if all the numbers are part of the same population, plus we can do mean tests. It's all the same :)
#Let's first recover the datasets we are gonna use. We are interested in 2012-2024. 
#First check we should do, does it make sense to assume index 20
#kbhc3r['purchase_price/sqm'][kbhc3r['Year_sold']==2012]
#plt.hist(kbhc3r['purchase_price/sqm'][kbhc3r['Year_sold']==2017], bins=int(np.sqrt(len(kbhc3r['purchase_price/sqm'][kbhc3r['Year_sold']==2017])).round()), color='salmon', alpha=0.7,histtype='step')
#It's not crazy to assume we have normal distributins, but it's clearly a matter of big numbers rather than something that comes for that.
#We then compute the intervals.
t_stat, p_value = stats.ttest_ind(kbhc3r['purchase_price/sqm'][kbhc3r['Year_sold']==2017], kbhc5r['purchase_price/sqm'][kbhc5r['Year_sold']==2017], equal_var=False)
print(t_stat,p_value)
#We see that p-values are pretty big. t-stats are also small. So yeah, fluctuations could be by chance, which implies that, indeed, the hypothesis is what we wanted.
#We compute now for several values the difference of means to check if there is statistical significance
def mcint(x,y,s_1,s_2,n_1,n_2,conf):
    point1=x-y-norm.ppf((1 + conf) / 2)*np.sqrt(s_1**2/n_1**2+s_2**2/n_2**2)
    point2=x-y+norm.ppf((1 + conf) / 2)*np.sqrt(s_1**2/n_1**2+s_2**2/n_2**2)
    print("[",point1,point2,"]")

# %%
mcint(kbhc3r_desc['mean'][kbhc3r_desc['Year_sold']==2023],kbhc4r_desc['mean'][kbhc4r_desc['Year_sold']==2023],kbhc3r_desc['std'][kbhc3r_desc['Year_sold']==2023],kbhc4r_desc['std'][kbhc4r_desc['Year_sold']==2023],kbhc3r_desc['count'][kbhc3r_desc['Year_sold']==2023],kbhc4r_desc['count'][kbhc4r_desc['Year_sold']==2023],0.95)
# %%
norm.ppf((1 + 0.95) / 2)
# %%
#We see that the test is negative. This is, with 95% confidence, we cannot say the means are different. We can try a different method with the chi^2 test
from scipy.stats import chi2_contingency
#We have to divide the prizes of the several houses again into intervals, and then compute the frequency tables
intervals_chi=[0,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,280000,300000]
kbhc3r_2023=kbhc3r[kbhc3r['Year_sold']==2023]
kbhc4r_2023=kbhc4r[kbhc4r['Year_sold']==2023]
kbhc3r_2023['Interval'] = pd.cut(kbhc3r_2023['purchase_price/sqm'], bins=intervals_chi, labels=False, include_lowest=True)
kbhc4r_2023['Interval'] = pd.cut(kbhc4r_2023['purchase_price/sqm'], bins=intervals_chi, labels=False, include_lowest=True)
freq3r_2023=kbhc3r_2023.groupby('Interval').size().reset_index(name='count')
freq4r_2023=kbhc4r_2023.groupby('Interval').size().reset_index(name='count')
interval_df_chi=pd.DataFrame({'Interval':range(len(intervals_chi)-1)})
freq3r_2023_full=pd.merge(interval_df_chi, freq3r_2023, on='Interval', how='left').fillna(0)
freq4r_2023_full=pd.merge(interval_df_chi, freq4r_2023, on='Interval', how='left').fillna(0)
# %%
# %%
chi2, p, dof, expected = chi2_contingency(table)
