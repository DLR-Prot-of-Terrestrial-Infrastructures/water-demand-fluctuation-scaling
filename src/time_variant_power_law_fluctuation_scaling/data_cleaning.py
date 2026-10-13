# Copyright (c) YYYY German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Bernhard Jonathan Sattler, Bernhard.sattler@dlr.de
# Date of creation: 2026-10-09

import pandas as pd
import numpy as np


def import_data(import_excel, all = True, number_rows = 2000):
    '''Data import, specifically for the BWDF data set.
    import_excel: docstring of the Excel containing the inflow data.
    all: True if all rows of the data shold be analyzed.
    number_rows: only relevant if all==False. number of first rows of the data that is analyzed.'''
    # Importing Data from the "Battle of Water Demand Forecasting (BWDF)”. 
    import_df = pd.read_excel(import_excel, index_col=0)

    # Setting the first column as index and foramtting to correct DateTime format
    import_df.index = pd.to_datetime(import_df.index, format='%d/%m/%Y %H:%M')

    if all:
        import_df = import_df
    else:
        import_df = import_df.iloc[:number_rows, :]

    return import_df

def clean_data(import_df, time_changes_strings_list = ["2021-03-28","2021-10-31","2022-03-27","2022-10-30","2023-03-26"]):

    '''Data cleaning, deletes all data of days that do not contain exactly the 24 hourly values to constitute a day.
    import_df: DataFrame containing the raw data.
    time_changes_strings_list: strings of the dates which were changes between daylight saving time and standard time in the years 2021 to 2023. These will be deleted from the DataFrame.'''
    # Delete duplicates. NOTE: this gets rid of the double values for the days of the shift from standard time to daylight savings time. Those days will be excluded later.
    import_no_duplicates_df = import_df.groupby(import_df.index).first()

    # Define the start and end dates
    start_date = import_no_duplicates_df.index.min()
    end_date = import_no_duplicates_df.index.max()

    # Generate the date range with hourly frequency
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')

    # Create the DataFrame using the time_index
    full_timeline_df = pd.DataFrame(index=time_index, columns= import_no_duplicates_df.columns)

    #insert data from import in completed DF
    full_timeline_df.update(import_no_duplicates_df)

    '''this section drops all values of days changing to or from daylight savings time'''

    time_changes_list = pd.to_datetime(time_changes_strings_list)

    full_timeline_no_daylight_saving_df = full_timeline_df.copy(deep=True)

    for date in time_changes_list:
        if date in full_timeline_df.index:
            full_timeline_no_daylight_saving_df.loc[date] = np.nan
        else:
            print("date does not exist: " + str(date))

    cleaned_df = pd.DataFrame(data="NaN", index=full_timeline_no_daylight_saving_df.index, columns= full_timeline_no_daylight_saving_df.columns)

    cleaned_mean_daily_demand_df = cleaned_df.resample('d').max()
    cleaned_full_timeline_no_daylight_saving_df = full_timeline_no_daylight_saving_df.copy(deep=True)

    for col in full_timeline_no_daylight_saving_df.columns:
        data = full_timeline_no_daylight_saving_df[col].copy(deep=True)

        mask = data.isna()

        #daily time series containing all days for which there was a NaN value present
        mask_daily = mask.resample('d').max()
        
        #daily averages. The resampling does not have to be on the mean, max or min would work just as well.
        data_daily = data.resample('d').mean()

        #filtered for only those days with complete data
        daily_filtered = data_daily[~mask_daily]

        #save in DataFrame columns
        cleaned_mean_daily_demand_df[col]=daily_filtered

        #Extract the date part from the hourly series index, i.e. a series where each day is indexed 24 duplicates of its date
        hourly_dates = full_timeline_no_daylight_saving_df.index.date

        # Map the daily values to each corresponding hour
        mask_hourly = mask_daily.reindex(hourly_dates).values    

        #filtered for only those days that are fully reported
        filtered_data= data[~mask_hourly]

        #save in DataFrame columns
        cleaned_full_timeline_no_daylight_saving_df[col] = filtered_data

        '''note: in theory this can be sped up by daily averaging over cleaned_full_timeline_no_daylight_saving_df.
        However, the chosen structure gives the oportunity to verify that the process worked by comparing the two time series.'''

    return cleaned_full_timeline_no_daylight_saving_df

def clean_wsnd_data(data):

    no_index = data.reset_index()

    # delete duplicate data
    no_exact_duplicates_df = no_index.drop_duplicates(
        subset=["Timestamp", "Flow (m3/hr)"],
        keep="first"
    )

    no_exact_duplicates_df = no_exact_duplicates_df.set_index("Timestamp")

    duplicate_timestamp = no_exact_duplicates_df.index.duplicated(keep=False)
    duplicate_value = no_exact_duplicates_df["Flow (m3/hr)"].duplicated(keep="first")

    # delete all but the first occurence of a value if timestamp and value exist multiple times within the df
    no_dupes = no_exact_duplicates_df[~(duplicate_timestamp & duplicate_value)]

    # resample to hourly time series
    hourly_avg = no_dupes.resample('h').mean()

    return hourly_avg