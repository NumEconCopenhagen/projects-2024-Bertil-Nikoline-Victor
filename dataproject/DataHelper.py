# Importing the datetime module
import datetime

# A wrapper for multiple APIs with a pandas interface
# Run the below pip-commmand in your terminal to use the pandas-datareader: 
# %pip install pandas-datareader

# Importing the pandas_datareader module
import pandas_datareader



#####################
#     U.S. Data     #
#####################

def inflation_annual_US(startDate = datetime.datetime(1900,1,1), endDate = datetime.datetime(2024,1,1)):
    """
    Retrieves annual inflation data for the United States within a specified date range.

    Parameters:
    startDate (datetime.datetime, optional): The start date for the inflation data retrieval. Defaults to January 1, 1900.
    endDate (datetime.datetime, optional): The end date for the inflation data retrieval. Defaults to January 1, 2024.

    Returns:
    pandas.DataFrame: A DataFrame containing annual inflation data for the United States.
    """
    # Getting the data from the API
    inflation = pandas_datareader.data.DataReader('FPCPITOTLZGUSA', 'fred', startDate, endDate)

    # Extract the year from the index and assign it back
    inflation.index = inflation.index.year

    # Rename the index to 'Year' and reset the index to make it a column
    inflation = inflation.rename_axis('Year').reset_index()

    # Rename column
    inflation = inflation.rename(columns={'FPCPITOTLZGUSA': 'Inflation_US'})

    return inflation

def unemployment_annual_US(startDate = datetime.datetime(1900,1,1), endDate = datetime.datetime(2024,1,1)):
    """
    Retrieves annual unemployment rate data for the United States.

    Parameters:
    startDate (datetime.datetime, optional): Start date for data retrieval. Defaults to January 1, 1900.
    endDate (datetime.datetime, optional): End date for data retrieval. Defaults to January 1, 2024.

    Returns:
    pandas.DataFrame: A DataFrame containing annual unemployment rate data for the United States.
    """    
    # Getting the data from the API
    unemployment_monthly = pandas_datareader.data.DataReader('UNRATE', 'fred', startDate, endDate)

    # Resample the data to annual frequency
    unemployment_annual = unemployment_monthly.resample('AS', convention='start').asfreq()

    # Extract the year from the index and assign it back
    unemployment_annual.index = unemployment_annual.index.year

    # Rename the index to 'Year' and reset the index to make it a column
    unemployment_annual = unemployment_annual.rename_axis('Year').reset_index()

    # Rename column
    unemployment_annual = unemployment_annual.rename(columns={'UNRATE': 'Unemployment_US'})

    return unemployment_annual



########################
#     Denmark Data     #
########################

# Importing the DstApi to fetch data from the API
# Run this in your terminal to use the DstAPi: %pip install git+https://github.com/alemartinello/dstapi
from dstapi import DstApi

# Import pandas
import pandas as pd


def inflation_annual_DK():
    """
    Fetches annual inflation data for Denmark.

    Returns:
    DataFrame: A pandas DataFrame containing annual inflation data for Denmark.
               The DataFrame has columns 'Year' and 'Inflation_DK' representing the year and
               corresponding inflation rate respectively.
    """
    # Initializing DstApi object
    inflation_dk = DstApi('PRIS9')

    # Define the parameters for the API call
    params = inflation_dk._define_base_params(language='en')
    params

    # Get the inflation data
    inflation_dk_api = inflation_dk.get_data(params=params)

    # Rename column
    inflation_dk_api = inflation_dk_api.rename(columns={'TID': 'Year'})

    # Delete the TYPE column
    del inflation_dk_api['TYPE']

    # Rename INDHOLD column to DK_Inflation
    inflation_dk_api = inflation_dk_api.rename(columns={'INDHOLD': 'Inflation_DK'})

    # Convert the column to integer type
    inflation_dk_api['Inflation_DK'] = pd.to_numeric(inflation_dk_api['Inflation_DK'], errors='coerce')

    return inflation_dk_api


def unemployment_annual_DK(startDate = datetime.datetime(1900,1,1), endDate = datetime.datetime(2024,1,1)):
    """
    Retrieves annual unemployment data for Denmark from FRED database.

    Parameters:
    startDate (datetime.datetime, optional): Start date of the data retrieval period. Defaults to January 1, 1900.
    endDate (datetime.datetime, optional): End date of the data retrieval period. Defaults to January 1, 2024.

    Returns:
    pandas.DataFrame: DataFrame containing annual unemployment data for Denmark with columns 'Year' and 'Unemployment_DK'.
    """
    # Getting unemployment data from FRED instead of DST because FRED has a longer timeseries
    # Note that there is missing data from 1965 until 1969 in the dataset

    # Getting the data from the API
    unemployment = pandas_datareader.data.DataReader('LRUN74TTDKA156S', 'fred', startDate, endDate)

    # Extract the year from the index and assign it back
    unemployment.index = unemployment.index.year

    # Rename the index to 'Year' and reset the index to make it a column
    unemployment = unemployment.rename_axis('Year').reset_index()

    # Rename column
    unemployment = unemployment.rename(columns={'LRUN74TTDKA156S': 'Unemployment_DK'})

    return unemployment



######################
#     Japan Data     #
######################

def inflation_annual_JPN(startDate = datetime.datetime(1900,1,1), endDate = datetime.datetime(2024,1,1)):
    """
    Retrieves annual inflation data for Japan from a specified start date to end date.

    Parameters:
    startDate (datetime.datetime, optional): Start date for retrieving data. Defaults to January 1, 1900.
    endDate (datetime.datetime, optional): End date for retrieving data. Defaults to January 1, 2024.

    Returns:
    pandas.DataFrame: A DataFrame containing annual inflation data for Japan with columns 'Year' and 'Inflation_JPN'.
    """
    # Getting the data from the API
    inflation = pandas_datareader.data.DataReader('FPCPITOTLZGJPN', 'fred', startDate, endDate)

    # Extract the year from the index and assign it back
    inflation.index = inflation.index.year

    # Rename the index to 'Year' and reset the index to make it a column
    inflation = inflation.rename_axis('Year').reset_index()

    # Rename column
    inflation = inflation.rename(columns={'FPCPITOTLZGJPN': 'Inflation_JPN'})

    return inflation

def unemployment_annual_JPN(startDate = datetime.datetime(1900,1,1), endDate = datetime.datetime(2024,1,1)):
    """
    Retrieves annual unemployment data for Japan.

    Parameters:
    startDate (datetime.datetime, optional): The start date for fetching the data. Defaults to January 1, 1900.
    endDate (datetime.datetime, optional): The end date for fetching the data. Defaults to January 1, 2024.

    Returns:
    pandas.DataFrame: A DataFrame containing annual unemployment data for Japan.
    """    
    # Getting the data from the API
    unemployment_monthly = pandas_datareader.data.DataReader('LRHUTTTTJPA156S', 'fred', startDate, endDate)

    # Resample the data to annual frequency
    unemployment_annual = unemployment_monthly.resample('AS', convention='start').asfreq()

    # Extract the year from the index and assign it back
    unemployment_annual.index = unemployment_annual.index.year

    # Rename the index to 'Year' and reset the index to make it a column
    unemployment_annual = unemployment_annual.rename_axis('Year').reset_index()

    # Rename column
    unemployment_annual = unemployment_annual.rename(columns={'LRHUTTTTJPA156S': 'Unemployment_JPN'})

    return unemployment_annual




###########################
#     Merged Datasets     #
###########################

def merged_data():
    """
    Merges multiple data sources containing inflation and unemployment rates for different countries.

    Returns:
    DataFrame: A pandas DataFrame containing merged data from multiple sources.

    Note:
    This function merges six different data sources for three countries (US, DK, JPN) with two data series 
    (inflation and unemployment rates) based on the 'Year' column using an outer merge. It then filters the 
    merged data to include only years between 1960 and 2022.
    """
    # Outer merge all of the six (3 countries x 2 data series) data sources
    data = pd.merge(inflation_annual_US(), unemployment_annual_US(), on='Year', how='outer')
    data = pd.merge(data, inflation_annual_DK(), on='Year', how='outer')
    data = pd.merge(data, unemployment_annual_DK(), on='Year', how='outer')
    data = pd.merge(data, inflation_annual_JPN(), on='Year', how='outer')
    data = pd.merge(data, unemployment_annual_JPN(), on='Year', how='outer')

    # Filter data for years before 1960 since most data is missing
    data = data[data['Year'] > 1959]

    # Filter data for years after 2022 since most data is missing
    data = data[data['Year'] <= 2022]

    return data
