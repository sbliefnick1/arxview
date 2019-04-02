#!/anaconda3/envs/arxview/bin python

from datetime import timedelta
import os
from urllib.parse import quote_plus
import urllib3
from urllib3.exceptions import InsecureRequestWarning

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import requests
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine, Column, Integer, VARCHAR, Date, Float, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config.get_arxview_config import fi_dm_ebi, user, password, connection_string, api_url

urllib3.disable_warnings(InsecureRequestWarning)

s = requests.Session()
s.auth = HTTPBasicAuth(user, password)


def query_data(method, parameters=None):
    # log in
    r = s.post(f'{api_url}/login.php', {'u': user, 'p': password}, verify=False)
    r.raise_for_status()

    # establish parameters
    params = {'format': 'json', 'bool': 1}

    if parameters:
        params = {**params, **parameters}

    # query the api
    r = s.get(f'{api_url}/{method}', params=params, verify=False)
    r.raise_for_status()

    df = pd.DataFrame.from_dict(r.json())

    # log out before returning dataframe
    r = s.get(f'{api_url}/logout.php', verify=False)
    r.raise_for_status()

    return df


def calculate_lin_reg(dataframe, asset_id, num_cols, str_cols, plot=False):
    # get non-NaN values of dataframe
    df = dataframe[dataframe.arx_asset_id == asset_id][['discovery_date'] + num_cols + str_cols]
    df = df[~np.isnan(df.array_reported_usage)]

    # get arrays of values for inputs
    dates = df.discovery_date.to_numpy()
    y = df.array_reported_usage.to_numpy()

    # convert dates to numbers
    x = mdates.date2num(dates)

    # fit trend line
    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit)

    # create base dataframe to merge into
    # one date from beginning to end of date range
    # plus one year into the future
    first_date = df.discovery_date.iloc[0]
    last_date = df.discovery_date.iloc[-1]
    date_range = pd.date_range(first_date,
                               last_date + timedelta(364),
                               periods=(last_date - first_date).days + 365)

    base = pd.DataFrame(date_range, columns=['date'])

    # calculate usage trend and predict future usage
    base['usage_trend'] = fit_fn(mdates.date2num(base.date))

    # merge dfs together
    merged = pd.merge(base, df, how='left', left_on='date', right_on='discovery_date')
    del merged['discovery_date']

    # optional plotting
    if plot:
        data = []
        for c in merged.columns:
            if merged[c].dtype == float:
                trace = go.Scatter(
                        x=merged.date,
                        y=merged[c],
                        name=c,
                        connectgaps=True
                        )

                data.append(trace)

            layout = {'title': merged.name.iloc[0],
                      'xaxis': {'title': 'Date'},
                      'yaxis': {'title': 'GB'}}

            fig = {'data': data, 'layout': layout}

            plotly.offline.plot(fig, filename=f'{merged.name.iloc[0]}.html')

    return merged


def main():
    # get data
    metrics = ['total_allocation', 'total_capacity', 'total_raw_capacity', 'array_reported_usage']
    cols = ['arx_asset_id', 'assettype_name', 'datacenter_name', 'name', 'vendor']
    arrays_growth_detail = query_data('getArrayGrowthDetail', parameters={'metrics': 'all'})

    # convert to appropriate dtypes
    arrays_growth_detail[metrics] = arrays_growth_detail[metrics].apply(pd.to_numeric)
    arrays_growth_detail['discovery_date'] = arrays_growth_detail['discovery_date'].apply(pd.to_datetime)

    # iterate over each asset to calculate usage_trend
    usage = pd.DataFrame()

    for a in arrays_growth_detail.arx_asset_id.unique():
        temp = calculate_lin_reg(arrays_growth_detail, a, metrics, cols, plot=False)
        usage = usage.append(temp, ignore_index=True)

    # fill in missing data
    usage[cols] = usage[cols].fillna(method='ffill')

    # create id column
    usage['id'] = usage.arx_asset_id + usage.date.dt.strftime('%Y%m%d')
    usage['id'] = usage['id'].apply(pd.to_numeric)

    # rename columns
    usage = usage.rename(columns={'name': 'asset_name',
                                  'date': 'asset_date',
                                  'assettype_name': 'asset_type_name'})

    # replace NaNs with None
    usage = usage.replace({np.nan: None})

    # begin sqlalchemy mapping
    if os.sys.platform == 'win32':
        ppw_engine = create_engine(connection_string)
    else:
        ppw_params = quote_plus('DRIVER={driver};'
                                'SERVER={server};'
                                'DATABASE={database};'
                                'UID={user};'
                                'PWD={password};'
                                'PORT={port};'
                                'TDS_Version={tds_version};'
                                .format(**fi_dm_ebi))
        ppw_engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(ppw_params))

    Base = declarative_base()

    class ArxviewArray(Base):
        __tablename__ = 'EBI_Arxview_Array'

        id = Column(BigInteger, primary_key=True, autoincrement=False)
        asset_date = Column(Date)
        usage_trend = Column(Float)
        total_allocation = Column(Float)
        total_capacity = Column(Float)
        total_raw_capacity = Column(Float)
        array_reported_usage = Column(Float)
        arx_asset_id = Column(Integer)
        asset_type_name = Column(VARCHAR(None))
        datacenter_name = Column(VARCHAR(None))
        asset_name = Column(VARCHAR(None))
        vendor = Column(VARCHAR(None))

    Session = sessionmaker(bind=ppw_engine)
    session = Session()

    Base.metadata.create_all(ppw_engine)

    session.bulk_update_mappings(ArxviewArray, usage.to_dict(orient='records'))
    session.commit()


if __name__ == '__main__':
    main()
    print('Successfully updated Arxview data')
