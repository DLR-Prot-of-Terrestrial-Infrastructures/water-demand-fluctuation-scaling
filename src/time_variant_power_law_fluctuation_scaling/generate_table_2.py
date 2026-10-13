# Copyright (c) YYYY German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Bernhard Jonathan Sattler, Bernhard.sattler@dlr.de
# Date of creation: 2026-10-09

import pandas as pd
import numpy as np
import h5py

def regime_lims(summary_statistic,
        regime_quantile_threshold,
        regime_identification_parameter,
        reporting_threshold,

        free_beta_result_path_dma,
        free_beta_result_path_wsnd
        ):
    '''Calculates the longest regimes for each time series.'''

    letters = ['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J', 'WSND']

    regime_df = pd.DataFrame(columns=np.arange(6))

    for letter in letters:

        if letter == 'WSND':
            with h5py.File(free_beta_result_path_wsnd, "r") as f_wsnd:

                match summary_statistic:
                    case 'mean':
                        group_label = 'mean posterior'
                    case 'MAP':
                        group_label = 'MAP'
                    case 'median':
                        group_label = 'median'
                        
                beta_in_regime = f_wsnd[letter][group_label][:,1]

                median = np.median(beta_in_regime)
                values = np.array([np.min(beta_in_regime), np.max(beta_in_regime)])
                regime_df.loc[letter, 0] = rf'\makecell{{${median:.1f};$\\${[f"{v:.1f}" for v in values]}$}}'


        else:
            with h5py.File(free_beta_result_path_dma, "r") as f_s:

                regime_dict = {}

                match summary_statistic:
                    case 'mean':
                        group_label = 'mean posterior'
                    case 'MAP':
                        group_label = 'MAP'
                    case 'median':
                        group_label = 'median'
                        
                log_alpha_ts = f_s[letter][group_label][:,0]
                beta_ts = f_s[letter][group_label][:,1]
                log_tau_ts = f_s[letter][group_label][:,2]

                time_labels = f_s[letter]['correct_index'][:]
                dt_index = np.unique(pd.to_datetime(time_labels).date)

                match regime_identification_parameter:
                    case 'alpha':
                        indicator = np.diff(log_alpha_ts)
                    case 'beta':
                        indicator = np.diff(beta_ts)
                    case 'tau':
                        indicator = np.diff(log_tau_ts)
                    case 'alpha-beta-spread':
                        indicator_temp = beta_ts - log_alpha_ts*np.mean(beta_ts)/np.mean(log_alpha_ts)
                        indicator = np.diff(indicator_temp)
                    case 'derivative_stability':
                        alpha_normalized = (log_alpha_ts-min(log_alpha_ts))/(max(log_alpha_ts)-min(log_alpha_ts))
                        beta_normalized = (beta_ts-min(beta_ts))/(max(beta_ts)-min(beta_ts))
                        tau_normalized = (log_tau_ts-min(log_tau_ts))/(max(log_tau_ts)-min(log_tau_ts))

                        magnitudes = np.sqrt(alpha_normalized**2+beta_normalized**2+tau_normalized**2)

                        indicator = np.diff(magnitudes)



                breaks = dt_index[:-1][np.abs(indicator)>np.quantile(np.abs(indicator), regime_quantile_threshold)]
                breaks = np.concatenate([breaks, np.array([dt_index[-1]])])
                section_lengths = breaks[1:] - breaks[:-1]
                section_lengths_in_days = [i.days for i in section_lengths]
                sorted_section_lengths = np.sort(section_lengths_in_days)
                lengths_of_regimes_above_threshold = sorted_section_lengths[sorted_section_lengths>=reporting_threshold]
                number_of_regimes_above_threshold = len(lengths_of_regimes_above_threshold)

                longest_lengths = np.argsort(section_lengths)[::-1][:number_of_regimes_above_threshold]

                starts = breaks[:-1][longest_lengths]
                regime_dict['starts'] = starts
                ends = breaks[1:][longest_lengths]
                regime_dict['ends'] = ends

                start_inds_raw, end_inds_raw = [],[]

                for start in starts:
                    start_inds_raw.append(np.arange(len(log_alpha_ts))[np.isin(dt_index,start)][0])
                for end in ends:
                    end_inds_raw.append(np.arange(len(log_alpha_ts))[np.isin(dt_index,end)][0])

                start_inds = np.array(start_inds_raw).astype(int)
                regime_dict['starts indeces'] = start_inds.astype(int)

                end_inds = np.array(end_inds_raw).astype(int)
                regime_dict['ends indeces'] = end_inds.astype(int)
                regime_dict['plot_mids'] = end_inds.astype(int)-1


                for i in range(len(regime_dict['starts indeces'])):

                    log_alpha_in_regime = log_alpha_ts[regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]
                    beta_in_regime = beta_ts[regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]
                    log_tau_in_regime = log_tau_ts[regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]

                    last_index = len(regime_df)

                    median = np.median(beta_in_regime)
                    values = np.array([np.min(beta_in_regime), np.max(beta_in_regime)])
                    regime_df.loc[letter, i] = rf'\makecell{{${median:.1f};$\\${[f"{v:.1f}" for v in values]}$}}'

    return regime_df
                    

def generate_csv(free_beta_result_path_dma,
            free_beta_result_path_wsnd,
            export_path,

            summary_statistic,
            regime_quantile_threshold,
            regime_identification_parameter,
            reporting_threshold,
            ):
    '''Generates LaTeX table for Table 2 in the result section.'''

    regime_variables_df = regime_lims(
            summary_statistic = summary_statistic,
            regime_quantile_threshold = regime_quantile_threshold,
            regime_identification_parameter = regime_identification_parameter,
            reporting_threshold = reporting_threshold,
            
            free_beta_result_path_dma = free_beta_result_path_dma,
            free_beta_result_path_wsnd = free_beta_result_path_wsnd)

    regime_variables_df = regime_variables_df.fillna('-')

    regime_variables_df.to_csv(export_path, sep = '&', lineterminator='\\\\\n')

    return '.csv generated'

def clean_csv(csv_path_input, csv_path_output):
    '''creates csv to match LaTeX code.'''

    with open(csv_path_input, encoding="utf-8") as f:
        text = f.read()
        text = text.replace("'", "").replace('"', "")

    with open(csv_path_output, "w", encoding="utf-8") as f:
        f.write(text)

    return '.csv cleaned.'

