import streamlit as st
import datetime
import altair as alt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.ticker as ticker
import geopandas as gp
import geopandas as gpd
import calmap
import folium
from folium.plugins import HeatMapWithTime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Dense, Activation, LeakyReLU
from keras import models
from keras.optimizers import RMSprop, Adam
import tensorflow as tf
from tensorflow.keras import layers

st.title("Covid-19 Basic Visualization and Prediction")
st.markdown(
"""
Collect data about COVID-19 and any related data. Here I just give several data sources like [COVID-19 Open Datasets](https://www.aminer.cn/data-covid19/?from=singlemessage)，[COVID-19 Epidemic Situation and Economic Research](http://cn.gtadata.com/#/datacenter/singletable/search?serId=16&databaseId=190). I adivse you find new data source for your research question. So, please feel free to get any data that related to your study.
- If you can download data directly, please list the url from which you get the data. Otherwise, please make a crawler with python by yourself to get the data.
- If you think it is necessary to preprocess data before jump into specific research question, please do some data precessing in this section. If you think it is better to conduct data clean in the subsequent sections, please do it later.
""")

#get data
confirm_US_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
confirm_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
death_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
death_US_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
recover_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
group = pd.read_excel('group.xlsx')
confirm = pd.read_csv(confirm_url)
confirm_US = pd.read_csv(confirm_US_url)
death = pd.read_csv(death_url)
death_US = pd.read_csv(death_US_url)
recover = pd.read_csv(recover_url)
confirm['group']=group
m=0
for i in confirm['group']:
    if i == 'Africa ':
        confirm['group'][m]='Africa'
    m+=1
group_death=[]
for i in death['Country/Region']:
    if i in confirm['Country/Region'].values:
        group_death.append(confirm[confirm['Country/Region']==i]['group'][max(confirm[confirm['Country/Region']==i]['group'].index)])
    else:
        print("not")
death['group']=group_death
group_recover=[]
for i in recover['Country/Region']:
    if i in confirm['Country/Region'].values:
        group_recover.append(confirm[confirm['Country/Region']==i]['group'][max(confirm[confirm['Country/Region']==i]['group'].index)])
    else:
        print("not")
recover['group']=group_recover

countries = sorted(np.unique(confirm['group']))
new = confirm.columns[len(confirm.columns)-2]
# Headings
st.header("Compare spread of Covid-19")
# Display data frame
st.subheader("Regions with most cases")
st.markdown("Filter countries by selecting below. Clear options to see all countries.")
def all():
    confirm_group=confirm.drop(columns=['Lat','Long']).groupby('group').sum()
    confirm_group=confirm_group['5/2/20']
    death_group=death.drop(columns=['Lat','Long']).groupby('group').sum()
    death_group=death_group['5/2/20']
    recover_group=recover.drop(columns=['Lat','Long']).groupby('group').sum()
    recover_group=recover_group['5/2/20']
    confirm_group.reset_index()
    death_group.reset_index()
    recover_group.reset_index()
    result=pd.merge(confirm_group, death_group, on='group')
    result=pd.merge(result, recover_group, on='group')
    result.rename(columns = {"5/2/20_x": "confirm", "5/2/20_y":"death","5/2/20":"recover"},  inplace=True)
    result['active']=result['confirm']-result['death']-result['recover']
    result['mortality rate']=result['death']/result['confirm']
    result['recovery rate']=result['recover']/result['confirm']
    return result.style.background_gradient(cmap='Reds',subset=['confirm'])\
            .background_gradient(cmap='Blues',subset=['death'])\
            .background_gradient(cmap='Greens',subset=['recover'])\
            .background_gradient(cmap='Purples',subset=['active'])\
            .background_gradient(cmap='Pastel1_r',subset=['mortality rate'])\
            .background_gradient(cmap='YlOrBr',subset=['recovery rate'])
show_all = all()
st.dataframe(show_all)

options_countries = st.multiselect("Regions", countries, default=['Asia', 'Europe'])

def format_diaplay_regions(countries):
    if len(options_countries) == 0:
        confirm_country1 = confirm.drop(columns=['Lat', 'Long', 'Province/State']).groupby('Country/Region').sum()
        confirm_country1 = confirm_country1[new]
        death_country1 = death.drop(columns=['Lat', 'Long', 'Province/State']).groupby('Country/Region').sum()
        death_country1 = death_country1[new]
        recover_country1 = recover.drop(columns=['Lat', 'Long', 'Province/State']).groupby('Country/Region').sum()
        recover_country1 = recover_country1[new]
        confirm_country1.reset_index()
        death_country1.reset_index()
        recover_country1.reset_index()
        result1 = pd.merge(confirm_country1, death_country1, on='Country/Region')
        result1 = pd.merge(result1, recover_country1, on='Country/Region')
        result1.rename(columns={"{}_x".format(new): "confirm", "{}_y".format(new): "death", "{}".format(new): "recover"},
                      inplace=True)
        result1['active'] = result1['confirm'] - result1['death'] - result1['recover']
        result1['mortality rate'] = result1['death'] / result1['confirm']
        result1['recovery rate'] = result1['recover'] / result1['confirm']
        result = result1.sort_values(by='confirm', ascending=False)
        return result.style.background_gradient(cmap='Reds', subset=['confirm']) \
            .background_gradient(cmap='Blues', subset=['death']) \
            .background_gradient(cmap='Greens', subset=['recover']) \
            .background_gradient(cmap='Purples', subset=['active']) \
            .background_gradient(cmap='Pastel1_r', subset=['mortality rate']) \
            .background_gradient(cmap='YlOrBr', subset=['recovery rate'])
    else:
         confirm_country = confirm[confirm['group'].isin(countries)].drop(columns=['Lat', 'Long', 'Province/State']).groupby('Country/Region').sum()
         confirm_country = confirm_country[new]
         death_country = death.drop(columns=['Lat', 'Long', 'Province/State']).groupby('Country/Region').sum()
         death_country = death_country[new]
         recover_country = recover.drop(columns=['Lat', 'Long', 'Province/State']).groupby('Country/Region').sum()
         recover_country = recover_country[new]
         confirm_country.reset_index()
         death_country.reset_index()
         recover_country.reset_index()
         result = pd.merge(confirm_country, death_country, on='Country/Region')
         result = pd.merge(result, recover_country, on='Country/Region')
         result.rename(columns={"{}_x".format(new): "confirm", "{}_y".format(new): "death", "{}".format(new): "recover"}, inplace=True)
         result['active'] = result['confirm'] - result['death'] - result['recover']
         result['mortality rate'] = result['death'] / result['confirm']
         result['recovery rate'] = result['recover'] / result['confirm']
         result = result.sort_values(by='confirm', ascending=False)
         return result.style.background_gradient(cmap='Reds', subset=['confirm']) \
             .background_gradient(cmap='Blues', subset=['death']) \
             .background_gradient(cmap='Greens', subset=['recover']) \
             .background_gradient(cmap='Purples', subset=['active']) \
             .background_gradient(cmap='Pastel1_r', subset=['mortality rate']) \
             .background_gradient(cmap='YlOrBr', subset=['recovery rate'])

disp_df = format_diaplay_regions(options_countries)

st.dataframe(disp_df, height=500)

confirm_con=confirm.groupby('Country/Region').sum()
death_con=death.groupby('Country/Region').sum()
recover_con=recover.groupby('Country/Region').sum()
confirm_con.drop(columns=['Lat','Long'],inplace=True)
death_con.drop(columns=['Lat','Long'],inplace=True)
recover_con.drop(columns=['Lat','Long'],inplace=True)

confirm_con['country']=confirm_con.index
confirm_con.reset_index(drop=True)
death_con['country']=death_con.index
death_con.reset_index(drop=True)
recover_con['country']=recover_con.index
recover_con.reset_index(drop=True)
group=[]
for i in confirm_con['country']:
    if i in confirm['Country/Region'].values:
        group.append(confirm[confirm['Country/Region']==i]['group'][max(confirm[confirm['Country/Region']==i]['group'].index)])
    else:
        print("not")
confirm_con['group']=group
group_death1=[]
for i in death_con['country']:
    if i in death['Country/Region'].values:
        group_death1.append(death[death['Country/Region']==i]['group'][max(death[death['Country/Region']==i]['group'].index)])
    else:
        print("not")
death_con['group']=group_death1
group_recover1=[]
for i in recover_con['country']:
    if i in recover['Country/Region'].values:
        group_recover1.append(recover[recover['Country/Region']==i]['group'][max(recover[recover['Country/Region']==i]['group'].index)])
    else:
        print("not")
recover_con['group']=group_recover1

colors = dict(zip(['Asia', 'South/Latin America', 'Africa', 'Europe', 'Arab States','Middle east', 'Oceania', 'North America'],
                 ['#DDA0DD','#F39B74','#F3F369','#D7F369','#B1F369','#69F3AF','#69F3E9','#69CCF3']))
group_lk=confirm.set_index('Country/Region')['group'].to_dict()


kind = ['confirm','death','recover']
option_kind = st.selectbox('Kind', kind)


try:
    video_file = open('bar_{}.mp4'.format(option_kind), 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
except:
    def draw_bar_confirm(date):
        test = confirm_con.loc[:, [date, 'country']]
        # 降序排列
        test = test.sort_values(by=date, ascending=False).head(20)
        test = test.sort_values(by=date)
        ax.clear()
        # 用世界范围的时间序列数据进行绘图
        ax.barh(test['country'], test[date], color=[colors[group_lk[x]] for x in test['country']])
        dx = int(test[date].max()) / 200
        # 设置数据显示
        for i, (value, country) in enumerate(zip(test[date], test['country'])):
            ax.text(value + dx, i, country, size=14, weight=600, ha='left', va='bottom')
            ax.text(value + dx, i - .25, f'{value:,.0f}', size=14, ha='left', va='baseline')
        # 设置表格中提示性的文字
        ax.text(1, 0.4, date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
        ax.text(0, 1.06, 'Population', transform=ax.transAxes, size=12, color='#777777')
        # 设置坐标轴和刻度格式
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='#777777', labelsize=12)
        ax.set_yticks([])
        ax.margins(0, 0.01)
        # 设置网格线
        ax.grid(which='major', axis='x', linestyle='-')
        ax.set_axisbelow(True)
        ax.text(0, 1.12, 'The number of people confirmed from 1.22 to {}'.format(new), transform=ax.transAxes, size=24,
                weight=600, ha='left')
        # 取消图标外框
        plt.box(False)


    fig, ax = plt.subplots(figsize=(28, 18))

    animator = animation.FuncAnimation(fig, draw_bar_confirm,
                                       frames=[x for x in confirm_con.columns.drop(['country', 'group'])])
    HTML(animator.to_jshtml())

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    animator.save(filename='bar_confirm.mp4', writer=writer)


    def draw_bar_death(date):
        test = death_con.loc[:, [date, 'country']]
        test = test.sort_values(by=date, ascending=False).head(20)
        test = test.sort_values(by=date)
        ax.clear()
        ax.barh(test['country'], test[date], color=[colors[group_lk[x]] for x in test['country']])
        dx = int(test[date].max()) / 200
        for i, (value, country) in enumerate(zip(test[date], test['country'])):
            ax.text(value + dx, i, country, size=14, weight=600, ha='left', va='bottom')
            ax.text(value + dx, i - .25, f'{value:,.0f}', size=14, ha='left', va='baseline')
        ax.text(1, 0.4, date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
        ax.text(0, 1.06, 'Population', transform=ax.transAxes, size=12, color='#777777')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='#777777', labelsize=12)
        ax.set_yticks([])
        ax.margins(0, 0.01)
        ax.grid(which='major', axis='x', linestyle='-')
        ax.set_axisbelow(True)
        ax.text(0, 1.12, 'The number of people death from 1/22 to {}'.format(new), transform=ax.transAxes, size=24,
                weight=600, ha='left')
        plt.box(False)


    fig, ax = plt.subplots(figsize=(28, 18))
    animator = animation.FuncAnimation(fig, draw_bar_death,
                                       frames=[x for x in death_con.columns.drop(['country', 'group'])])
    HTML(animator.to_jshtml())

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    animator.save(filename='bar_death.mp4', writer=writer)


    def draw_bar_recover(date):
        test = recover_con.loc[:, [date, 'country']]
        test = test.sort_values(by=date, ascending=False).head(20)
        test = test.sort_values(by=date)
        ax.clear()
        ax.barh(test['country'], test[date], color=[colors[group_lk[x]] for x in test['country']])
        dx = int(test[date].max()) / 200
        for i, (value, country) in enumerate(zip(test[date], test['country'])):
            ax.text(value + dx, i, country, size=14, weight=600, ha='left', va='bottom')
            ax.text(value + dx, i - .25, f'{value:,.0f}', size=14, ha='left', va='baseline')
        ax.text(1, 0.4, date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
        ax.text(0, 1.06, 'Population', transform=ax.transAxes, size=12, color='#777777')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='#777777', labelsize=12)
        ax.set_yticks([])
        ax.margins(0, 0.01)
        ax.grid(which='major', axis='x', linestyle='-')
        ax.set_axisbelow(True)
        ax.text(0, 1.12, 'The number of people recovered from 1/22 to {}'.format(new), transform=ax.transAxes, size=24,
                weight=600, ha='left')
        plt.box(False)


    fig, ax = plt.subplots(figsize=(28, 18))
    animator = animation.FuncAnimation(fig, draw_bar_recover,
                                       frames=[x for x in recover_con.columns.drop(['country', 'group'])])
    HTML(animator.to_jshtml())

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    animator.save(filename='bar_recover.mp4', writer=writer)

    video_file = open('bar_{}.mp4'.format(kind), 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

st.subheader("Case Map")
st.markdown("You can view the trend of infection changes during covid-19 through the timeline below.")

world = gp.read_file(
    gp.datasets.get_path('naturalearth_lowres')
)

time = st.slider('Select the day frome 1/22/20',1,len(confirm.columns.drop(['Province/State','Country/Region','Lat','Long','group']))-1)
confirm.loc[confirm['Country/Region']=='US', 'Country/Region'] = 'United States of America'
confirm.loc[confirm['Country/Region']=='Western Sahara', 'Country/Region'] = 'W. Sahara'
confirm.loc[confirm['Country/Region']=='Congo (Brazzaville)', 'Country/Region'] = 'Dem. Rep. Congo'
confirm.loc[confirm['Country/Region']=='Congo (Kinshasa)', 'Country/Region'] = 'Dem. Rep. Congo'
confirm.loc[confirm['Country/Region']=='Dominican Republic', 'Country/Region'] = 'Dominican Rep'
confirm.loc[confirm['Country/Region']=="Cote d'Ivoire'", 'Country/Region'] = "Côte d'Ivoire'"
confirm.loc[confirm['Country/Region']=='Central African Republic', 'Country/Region'] = 'Central African Rep'
confirm.loc[confirm['Country/Region']=='Guinea', 'Country/Region'] = 'Eq. Guinea'
confirm.loc[confirm['Country/Region']=='Eswatini', 'Country/Region'] = 'eSwatini'
confirm.loc[confirm['Country/Region']=='Congo (Kinshasa)', 'Country/Region'] = 'Dem. Rep. Congo'
confirm.loc[confirm['Country/Region']=='Korea, South', 'Country/Region'] = 'South Korea'
confirm.loc[confirm['Country/Region']=='Taiwan*', 'Country/Region'] = 'Taiwan'
confirm.loc[confirm['Country/Region']=='Somalia', 'Country/Region'] = 'Somaliland'
confirm.loc[confirm['Country/Region']=='Bosnia and Herzegovina', 'Country/Region'] = 'Bosnia and Herz'
confirm.loc[confirm['Country/Region']=='North Macedonia', 'Country/Region'] = 'Bosnia and Herz'
confirm.loc[confirm['Country/Region']=='Somalia', 'Country/Region'] = 'Somaliland'
confirm.loc[confirm['Country/Region']=='South Sudan', 'Country/Region'] = 'S. Sudan'


def world_map(day):
    plt.figure(figsize=(16, 9))
    confirm_country = confirm.drop(columns=['Lat', 'Long', 'Province/State']).groupby('Country/Region').sum()
    confirm_country1 = confirm_country.reset_index()
    date = confirm_country1.columns[day]
    data_geod = gp.GeoDataFrame(confirm_country1.loc[:, [date, 'Country/Region']])  ##将data转换为geopandas.DataFrame
    world_ = world.rename(index=str, columns={'name': 'Country/Region'})  # 修改列名
    da_merge = data_geod.merge(world_, how='left')  # 数据合并
    fig, ax = plt.subplots(figsize=(12, 12))
    da_merge.plot(ax = ax,column=date, k=4, cmap=plt.cm.Reds,edgecolor='k',lw=0.2)
    ax.axis('off')
    plt.title('map of {}'.format(date), size=20)
    plt.box(False)
    plt.show()
st.pyplot(world_map(time))

st.subheader("The following line chart shows the number of confirmed diagnoses, deaths and recovery in various countries since the recent epidemic:")
st.markdown("You can select the country you want to view by selecting.")


line_country = ['global','Africa', 'Arab States', 'Asia', 'Europe', 'Middle east', 'North America', 'Oceania', 'South/Latin America']
option_region = st.selectbox('Select a Region', line_country)
def line(region):
    if(region=='global'):
        plt.figure(figsize=(16, 9))
        # marker为设置各数据点的标识，markerfacecolor设置点内颜色
        last=confirm_con.index[len(confirm_con)-1]
        plt.plot(range(0, len(confirm.columns.drop(['Province/State','Country/Region','Lat','Long','group']))), confirm_con.drop(columns=['country', 'group']).cumsum().loc[last].values,
                 marker='o', markerfacecolor='#ffffff', linewidth=3)
        plt.plot(range(0, len(confirm.columns.drop(['Province/State','Country/Region','Lat','Long','group']))), death_con.drop(columns=['country', 'group']).cumsum().loc[last].values,
                 marker='o', markerfacecolor='#ffffff', linewidth=3)
        plt.plot(range(0, len(confirm.columns.drop(['Province/State','Country/Region','Lat','Long','group']))), recover_con.drop(columns=['country', 'group']).cumsum().loc[last].values,
                 marker='o', markerfacecolor='#ffffff', linewidth=3)
        # 填充颜色，alpha为设置颜色透明度
        plt.fill_between(confirm_con.drop(columns=['country', 'group']).columns,
                         confirm_con.drop(columns=['country', 'group']).cumsum().loc[last].values, alpha=0.2)
        plt.fill_between(death_con.drop(columns=['country', 'group']).columns,
                         death_con.drop(columns=['country', 'group']).cumsum().loc[last].values, alpha=0.2)
        plt.fill_between(recover_con.drop(columns=['country', 'group']).columns,
                         recover_con.drop(columns=['country', 'group']).cumsum().loc[last].values, alpha=0.2)
        plt.title('Global Confirmed, Death and Recover', size=30)
        plt.xlabel('Days since 1/22/2020', size=30)
        plt.ylabel('the Amount of People', size=30)
        # 设置图例
        plt.legend(['Confirmed', 'Death', 'Recover'], prop={'size': 20})
        plt.xticks([0, 20, 40, 60, 80, 100], size=20)
        plt.grid(which='major', linestyle='-.')
        plt.yticks(size=20)
        plt.margins(0.03)
        plt.box(False)
    else:
        confirm_country = confirm.drop(columns=['Lat', 'Long', 'Province/State']).groupby('group').sum()
        death_country = death.drop(columns=['Lat', 'Long', 'Province/State']).groupby('group').sum()
        recover_country = recover.drop(columns=['Lat', 'Long', 'Province/State']).groupby('group').sum()
        plt.figure(figsize=(16,9))
        plt.plot(confirm_country.loc[region].index,confirm_country.loc[region].values,marker='o',markerfacecolor='#ffffff',linewidth=3)
        plt.plot(confirm_country.loc[region].index,death_country.loc[region].values,marker='o',markerfacecolor='#ffffff',linewidth=3)
        plt.plot(confirm_country.loc[region].index,recover_country.loc[region].values,marker='o',markerfacecolor='#ffffff',linewidth=3)
        plt.fill_between(confirm_country.loc[region].index,confirm_country.loc[region].values,alpha=0.2)
        plt.fill_between(death_country.loc[region].index,death_country.loc[region].values,alpha=0.2)
        plt.fill_between(recover_country.loc[region].index,recover_country.loc[region].values,alpha=0.2)
        plt.title('{} Confirmed, Death and Recover'.format(region),size=30)
        plt.xlabel('Days since 1/22/2020',size=30)
        plt.ylabel('the Amount of People',size=30)
        plt.legend(['Confirmed','Death','Recover'],prop={'size':20})
        plt.xticks([0,20,40,60,80,100],size=20)
        plt.grid(which='major',linestyle='-.')
        plt.yticks(size=20)
        plt.margins(0.03)
        plt.box(False)

st.pyplot(line(option_region))

confirm_case=confirm_con.drop(columns=['country','group'])
death_case=death_con.drop(columns=['country','group'])
recover_case=recover_con.drop(columns=['country','group'])
linecase = st.selectbox('Select a Country to see details',confirm_case.index)
def line_case(country):
    plt.figure(figsize=(16, 9))
    plt.plot(confirm_case.loc[country].index, confirm_case.loc[country].values, marker='o', markerfacecolor='#ffffff', linewidth=3)
    plt.plot(death_case.loc[country].index, death_case.loc[country].values, marker='o', markerfacecolor='#ffffff', linewidth=3)
    plt.plot(recover_case.loc[country].index, recover_case.loc[country].values, marker='o', markerfacecolor='#ffffff', linewidth=3)
    plt.fill_between(confirm_case.loc[country].index, confirm_case.loc[country].values, alpha=0.2)
    plt.fill_between(death_case.loc[country].index, death_case.loc[country].values, alpha=0.2)
    plt.fill_between(recover_case.loc[country].index, recover_case.loc[country].values, alpha=0.2)
    plt.title('{} Confirmed, Death and Recover'.format(country), size=30)
    plt.xlabel('Days since 1/22/2020', size=30)
    plt.ylabel('the Amount of People', size=30)
    plt.legend(['Confirmed', 'Death', 'Recover'], prop={'size': 20})
    plt.xticks([0, 20, 40, 60, 80, 100], size=20)
    plt.grid(which='major', linestyle='-.')
    plt.yticks(size=20)
    plt.margins(0.03)
    plt.box(False)

st.pyplot(line_case(linecase))

case = sorted(np.unique(confirm_con['country']))
options_cases = st.multiselect("Confirm Cases", case, default=['China', 'Italy','US'])



def country_confirm(con):
    plt.figure(figsize=(16, 9))
    for i in con:
        case_i = confirm_con.loc[i].drop(['country', 'group'])
        plt.plot(case_i.index, case_i.values, marker='o', markerfacecolor='#ffffff', linewidth=3)
    plt.title('Confirmed of Cases', size=30)
    plt.xlabel('Days since 1/22/2020', size=30)
    plt.ylabel('the Amount of People', size=30)
    plt.legend(con, prop={'size': 20})
    plt.xticks([0, 20, 40, 60, 80, 100], size=20)
    plt.grid(which='major', linestyle='-.')
    plt.yticks(size=20)
    plt.margins(0.03)
    plt.box(False)

def country_death(con):
    plt.figure(figsize=(16, 9))
    for i in con:
        case_i = death_con.loc[i].drop(['country', 'group'])
        plt.plot(case_i.index, case_i.values, marker='o', markerfacecolor='#ffffff', linewidth=3)
    plt.title('Confirmed of Cases', size=30)
    plt.xlabel('Days since 1/22/2020', size=30)
    plt.ylabel('the Amount of People', size=30)
    plt.legend(con, prop={'size': 20})
    plt.xticks([0, 20, 40, 60, 80, 100], size=20)
    plt.grid(which='major', linestyle='-.')
    plt.yticks(size=20)
    plt.margins(0.03)
    plt.box(False)

def country_recover(con):
    plt.figure(figsize=(16, 9))
    for i in con:
        case_i = recover_con.loc[i].drop(['country', 'group'])
        plt.plot(case_i.index, case_i.values, marker='o', markerfacecolor='#ffffff', linewidth=3)
    plt.title('Confirmed of Cases', size=30)
    plt.xlabel('Days since 1/22/2020', size=30)
    plt.ylabel('the Amount of People', size=30)
    plt.legend(con, prop={'size': 20})
    plt.xticks([0, 20, 40, 60, 80, 100], size=20)
    plt.grid(which='major', linestyle='-.')
    plt.yticks(size=20)
    plt.margins(0.03)
    plt.box(False)

st.pyplot(country_confirm(options_cases))
st.pyplot(country_death(options_cases))
st.pyplot(country_recover(options_cases))

pre_day = st.number_input('The day you predict',min_value=1,max_value=20)
pre_degree = st.number_input('The degree you choose',min_value=1,max_value=20,value=5)
def pred(day,degree):
    quadratic_featurizer = PolynomialFeatures(degree=degree)
    X_train_quadratic = quadratic_featurizer.fit_transform(np.arange(1, len(confirm.columns.drop(['Province/State','Country/Region','Lat','Long','group']))+1).reshape(-1, 1))
    regressor_model = LinearRegression()
    last = confirm_con.index[len(confirm_con)-1]
    regressor_model.fit(X_train_quadratic,confirm_con.drop(columns=['country', 'group']).cumsum().loc[last].values)

    X_test_quadratic = quadratic_featurizer.fit_transform(np.arange(1, len(confirm.columns.drop(['Province/State','Country/Region','Lat','Long','group']))+day+1).reshape(-1, 1))
    pre = regressor_model.predict(X_test_quadratic)
    plt.figure(figsize=(16, 9))
    plt.plot(range(0, len(confirm.columns.drop(['Province/State','Country/Region','Lat','Long','group']))), confirm_con.drop(columns=['country', 'group']).cumsum().loc['Zimbabwe'].values,
                marker='o', markerfacecolor='#ffffff', linewidth=3)
    plt.plot(range(0, len(confirm.columns.drop(['Province/State','Country/Region','Lat','Long','group']))+day), pre, '-.', markerfacecolor='#ffffff', linewidth=3)
    plt.title('Global Confirmed:Actual and Predict', size=30)
    plt.xlabel('Days since 1/22/2020', size=30)
    plt.ylabel('Confirmed Amount of People', size=30)
    plt.legend(['Actual', 'Predict'], prop={'size': 20})
    plt.xticks([0, 20, 40, 60, 80, 100], size=20)
    plt.grid(which='major', linestyle='-.')
    plt.yticks(size=20)
    plt.margins(0.03)
    plt.box(False)


st.pyplot(pred(pre_day,pre_degree))

st.markdown("In the current epidemic situation, we must always pay attention to protection, wash hands frequently, and jointly resist COVID-19")