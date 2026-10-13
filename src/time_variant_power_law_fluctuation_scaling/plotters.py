# Copyright (c) YYYY German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Bernhard Jonathan Sattler, Bernhard.sattler@dlr.de
# Date of creation: 2026-10-09

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import h5py

from data_cleaning import import_data, clean_data, clean_wsnd_data

plt.style.use('dlr_style.mplstyle')

def single_representative_day(file_type):

    '''This plots the water demand time series for a single day to illustrate how the data looks for the graphical abstract.
    1st September 2022 in the DMA D was chosen for illustration purposes.'''

    df = import_data('Inflow_Data_4.xlsx')

    df_clean = clean_data(df)

    # WSND data import
    raw_df=pd.read_csv('inflow.csv', index_col='Timestamp', date_format = "%m/%d/%Y %H:%M")

    # WSND cleaning
    hourly_avg = clean_wsnd_data(raw_df)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    italy_drop_year = df_clean.loc['2022-01-01':'2022-12-31'].copy()
    italy_drop_year.index = italy_drop_year.index.map(lambda x: x.replace(year=2000))

    wsnd_drop_year = hourly_avg.copy()
    wsnd_drop_year.index = wsnd_drop_year.index.map(lambda x: x.replace(year=2000))

    demand_array = italy_drop_year['DMA D (L/s)'].loc['2000-09-01':'2000-09-01'].values
    ax.plot(np.arange(len(demand_array)), demand_array, color = 'black')
    ax.plot(np.arange(len(demand_array)), np.mean(demand_array)*np.ones_like(demand_array), color = 'red')
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.set_ylabel('WATER DEMAND in L/s')
    ax.set_xlabel('TIME in h')
    #ax.set_xticks(np.arange(7))
    ax.set_yticks([])

    if file_type == 'png':
        plt.savefig('single_day.png')
    elif file_type == 'svg':
        plt.savefig('single_day.svg', transparent=True)
    plt.close(fig)

def plot_representative_input_data(file_type):
    '''This plots one week of data for all time series.
    As the WSND data only covers September, the first week of September was chosen.
    As the WSND data covers September 2023 and BWDF covers September only for 2021 and 2022, the presented data is chosen from Sep. 2023 fpr the WSND data and Sep. 2022 for the BWDF data.'''

    df = import_data('Inflow_Data_4.xlsx')

    df_clean = clean_data(df)

    # WSND data import
    raw_df=pd.read_csv('inflow.csv', index_col='Timestamp', date_format = "%m/%d/%Y %H:%M")

    # WSND cleaning
    hourly_avg = clean_wsnd_data(raw_df)

    fig, ax = plt.subplots()

    italy_drop_year = df_clean.loc['2022-01-01':'2022-12-31'].copy()
    italy_drop_year.index = italy_drop_year.index.map(lambda x: x.replace(year=2000))

    wsnd_drop_year = hourly_avg.copy()
    wsnd_drop_year.index = wsnd_drop_year.index.map(lambda x: x.replace(year=2000))


    ax.plot(italy_drop_year[[ 'DMA B (L/s)', 'DMA F (L/s)', 'DMA G (L/s)', 'DMA H (L/s)', 'DMA I (L/s)', 'DMA J (L/s)']].loc['2000-09-01':'2000-09-08'], color = 'black', alpha = 0.2)
    ax.plot(italy_drop_year['DMA C (L/s)'].loc['2000-09-01':'2000-09-08'], label = 'DMA C', color = 'black')
    ax.plot(italy_drop_year['DMA D (L/s)'].loc['2000-09-01':'2000-09-08'], label = 'DMA D', color = 'black', linestyle = ':')
    ax.plot(italy_drop_year['DMA E (L/s)'].loc['2000-09-01':'2000-09-08'], label = 'DMA E', color = 'black', linestyle = '--')
    ax.plot((1/3.6)*wsnd_drop_year.loc['2000-09-01':'2000-09-08'], label = 'WSND data', color = 'black', linestyle = '-.')
    ax.plot(italy_drop_year[['DMA A (L/s)']].loc['2000-09-01':'2000-09-08'], color = 'black', alpha = 0.1, label = 'other DMAs')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d. %b'))
    ax.set_ylabel('WATER DEMAND in L/s')
    ax.set_xlabel('DATE')

    if file_type == 'png':
        plt.savefig('exemplary_data.png')
    elif file_type == 'svg':
        plt.savefig('exemplary_data.svg')
    plt.close(fig)

def plot_dmas_vertical_only_beta(free_beta_result_path,
                        quadratic_result_path,
                        file_prefix,
                        file_type,
                        summary_statistic,
                        regime_quantile_threshold,
                        regime_identification_parameter,
                        no_displayed_regimes,
                        logarithmic_scatter,
                        plot_beta_lim,
                        plot_ribbon,
                        regime_padding):
    '''Plots simplified version of the result plots found in the graphical abstract.
    In the graphical abstract, the plot for DMA C was chosen.'''

    # Colors from the currently active style
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    letters = ['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J']

    upper_ylim = plot_beta_lim

    for letter in letters:
        with h5py.File(free_beta_result_path, "r") as f_s:

            regime_dict = {}

            fig, axs = plt.subplots(2, 1, figsize=(8, 4))

            ax1 = axs[0]
            ax3 = axs[1]
            
            mean = f_s[letter]['mean'][:]
            var = f_s[letter]['variance'][:]

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
            longest_lengths = np.argsort(section_lengths)[::-1][:no_displayed_regimes]

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

            # ---------- 1st Plot -----------------------------

            _beta_plot, = ax1.plot(dt_index, beta_ts, c = 'black', label = r'$\beta$')
            ax1.set_ylim(0, upper_ylim)
            ax1.set_ylabel(r'$\beta$')
            v_line_colors_non_tiled = np.array([colors[k] for k in range(no_displayed_regimes)])
            v_line_colors = np.tile(v_line_colors_non_tiled, 2)
            ax1.vlines(dt_index[[start_inds, end_inds]], 0, upper_ylim, v_line_colors)

            for i in range(len(regime_dict['starts indeces'])):
                ax1.fill_betweenx([0,upper_ylim], dt_index[start_inds[i]], dt_index[end_inds[i]], color = v_line_colors[i], alpha = 0.2)
            
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b. %y'))

            handles = [_beta_plot]
            labels = [h.get_label() for h in handles]

            ax1.legend(handles, labels)
            ax1.set_xlabel('DATE')
            ax1.tick_params(labelbottom=True)


            # ---------- 2nd Plot -----------------------------
            with h5py.File(quadratic_result_path, "r") as f_q:
                
                time_labels = f_s[letter]['correct_index'][:]
                dt_index = np.unique(pd.to_datetime(time_labels).date)

                # ------ 3rd plot -----------

                mask = np.ones(len(f_s[letter]['mean']), dtype=bool)

                for i in range(len(regime_dict['starts indeces'])):
                    mask[regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]] = False

                mean_raw = f_s[letter]['mean']
                var_raw = f_s[letter]['variance']
                
                ax3.set_ylim(min(var_raw), max(var_raw))

                mean = mean_raw[mask]
                var = var_raw[mask]

                ax3.scatter(mean, var, c= 'grey', s = 1, label = 'days not in regimes')

                #ax3.set_title(r'$DAILY  m$ - $s^2$ RELATION')

                markers = ['x', '<', 'o', '>', '^', 'v', '3']

                for i in range(len(regime_dict['starts indeces'])):

                    mean = f_s[letter]['mean'][regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]
                    var = f_s[letter]['variance'][regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]

                    if regime_padding ==0:
                        upper_regime_padding = -len(mean)
                    
                    x_array = np.linspace(min(mean), max(mean), 50)

                    if plot_ribbon:
                        plot_indeces = np.arange(start = regime_dict['starts indeces'][i], stop = regime_dict['ends indeces'][i])

                        for j in plot_indeces:
                            
                            plot_alpha = log_alpha_ts[j]
                            plot_beta = beta_ts[j]
                            y_array = np.exp(plot_alpha)*x_array**plot_beta
                            ax3.plot( x_array, y_array, color = 'grey', alpha = 0.2)

                    plot_mid = regime_dict['plot_mids'][i]

                    if i == 0:
                        beta_free_label = r'$s^2 = \alpha \cdot m^\beta, \beta>2$'
                        quadratic_label = r'$s^2 = \alpha \cdot m^2$'
                    else:
                        beta_free_label = ''
                        quadratic_label = ''

                    plot_alpha = log_alpha_ts[plot_mid]
                    plot_beta = beta_ts[plot_mid]
                    y_array = np.exp(plot_alpha)*x_array**plot_beta
                    ax3.plot( x_array, y_array, label = beta_free_label, c='black')# c= colors[i])

                    sc = ax3.scatter(mean[regime_padding:-upper_regime_padding], var[regime_padding:-upper_regime_padding], s=30, label = f"{regime_dict['starts'][i].strftime('%d %b. %y')} - {regime_dict['ends'][i].strftime('%d %b. %y')}", marker = markers[i]) # c= colors
                                        

                    offset_alpha = np.exp(log_tau_ts)/(np.cos(np.arctan(beta_ts)))
                    offset_x = np.exp(log_tau_ts[plot_mid])/(np.sin(np.arctan(beta_ts[plot_mid])))

                    x_upper_min = np.exp(np.log(min(mean))-offset_x)
                    x_upper_max = np.exp(np.log(max(mean))-offset_x)

                    x_lower_min = np.exp(np.log(min(mean))+offset_x)
                    x_lower_max = np.exp(np.log(max(mean))+offset_x)

                    '''square TL comparison model'''
                    beta_ts_quad = f_q[letter][group_label][:,1]
                    
                    log_alpha_ts_quad = f_q[letter][group_label][:,0]

                    y_array = np.exp(log_alpha_ts_quad[plot_mid])*x_array**beta_ts_quad[plot_mid]
                    ax3.plot( x_array, y_array, label = quadratic_label, linestyle = '--', c='black')#c= colors[i])

                ax3.set_ylabel(r'$s^2$ in $L^2/s^2$')
                ax3.set_xlabel(r'$m$ in $L/s$')

                x_plot_min_max_diff = max(mean_raw)- min(mean_raw)
                lower_x_padding = 0.05*x_plot_min_max_diff
                upper_x_padding = np.exp(1.1)*x_plot_min_max_diff

                ax3.set_xlim(min(mean_raw)-lower_x_padding, max(mean_raw)+upper_x_padding)

                if logarithmic_scatter:
                    ax3.set_xscale('log')
                    ax3.set_yscale('log')
                    
                ax3.legend(loc = 'lower right')
                plt.tight_layout()
                #plt.setp(ax1.get_xticklabels(), visible=False)
                fig.set_constrained_layout_pads(hspace=0.02)

                if file_type == 'png':
                    plt.savefig(f'{file_prefix}{letter}.png', bbox_inches='tight')
                elif file_type == 'svg':
                    plt.savefig(f'{file_prefix}{letter}.svg', bbox_inches='tight', transparent=True)
                plt.close(fig)

def plot_dmas_vertical(free_beta_result_path,
                        quadratic_result_path,
                        file_prefix,
                        file_type,
                        summary_statistic,
                        credible_interval,
                        ci_tau_multiplier,
                        regime_quantile_threshold,
                        regime_identification_parameter,
                        no_displayed_regimes,
                        logarithmic_scatter,
                        plot_beta_lim,
                        plot_ribbon,
                        regime_padding):

    '''This plots the full result plots for the BWDF data found in the paper.'''

    # Colors from the currently active style
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    letters = ['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J']

    upper_ylim = plot_beta_lim

    for letter in letters:
        with h5py.File(free_beta_result_path, "r") as f_s:

            regime_dict = {}

            fig, axs = plt.subplots(3, 1, figsize=(10, 12))

            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]
            
            mean = f_s[letter]['mean'][:]
            var = f_s[letter]['variance'][:]

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

            match credible_interval:
                case '90 %':
                    log_alpha_lower_ci = f_s[letter]['5%'][:,0]
                    log_alpha_upper_ci = f_s[letter]['95%'][:,0]

                    beta_lower_ci = f_s[letter]['5%'][:,1]
                    beta_upper_ci = f_s[letter]['95%'][:,1]

                    log_tau_lower_ci = f_s[letter]['5%'][:,2]
                    log_tau_upper_ci = f_s[letter]['95%'][:,2]

                case '50 %':
                    log_alpha_lower_ci = f_s[letter]['25%'][:,0]
                    log_alpha_upper_ci = f_s[letter]['75%'][:,0]

                    beta_lower_ci = f_s[letter]['25%'][:,1]
                    beta_upper_ci = f_s[letter]['75%'][:,1]

                    log_tau_lower_ci = f_s[letter]['25%'][:,2]
                    log_tau_upper_ci = f_s[letter]['75%'][:,2]

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
            longest_lengths = np.argsort(section_lengths)[::-1][:no_displayed_regimes]

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

            # ---------- 1st Plot -----------------------------

            _beta_plot, = ax1.plot(dt_index, beta_ts, c = 'black', label = r'$\beta$')
            ci_b_plot, = ax1.plot(dt_index, beta_lower_ci, c = 'black', linestyle =':', label = fr'{credible_interval} CI $\beta$')
            ax1.plot(dt_index, beta_upper_ci, c = 'black', linestyle =':')
            ax1.set_ylim(0, upper_ylim)
            ax1.set_ylabel(r'$\beta$')
            v_line_colors_non_tiled = np.array([colors[k] for k in range(no_displayed_regimes)])
            v_line_colors = np.tile(v_line_colors_non_tiled, 2)
            ax1.vlines(dt_index[[start_inds, end_inds]], 0, upper_ylim, v_line_colors)

            for i in range(len(regime_dict['starts indeces'])):
                ax1.fill_betweenx([0,upper_ylim], dt_index[start_inds[i]], dt_index[end_inds[i]], color = v_line_colors[i], alpha = 0.2)
            
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b. %y'))

            ax_1_b = ax1.twinx()
            log_alpha_plot, = ax_1_b.plot(dt_index, log_alpha_ts, c = 'grey', linestyle = '--', label = r'$\mathrm{log(}\alpha\mathrm{)}$')
            ax_1_b.plot(dt_index, log_alpha_lower_ci, c = 'grey', linestyle = ':')
            ci_a_plot, = ax_1_b.plot(dt_index, log_alpha_upper_ci, c = 'grey', linestyle = ':', label = fr'{credible_interval} CI $\alpha$')
            ax_1_b.set_ylim(0, upper_ylim*(np.mean(log_alpha_ts)/np.mean(beta_ts)))
            ax_1_b.set_ylabel(r'$\mathrm{log(} \alpha \mathrm{)}$')

            handles = [_beta_plot, ci_b_plot, log_alpha_plot, ci_a_plot]
            labels = [h.get_label() for h in handles]

            ax_1_b.spines["right"].set_visible(True)
            ax1.legend(handles, labels)
            ax1.tick_params(labelbottom=False)


            # ---------- 2nd Plot -----------------------------
            with h5py.File(quadratic_result_path, "r") as f_q:
                
                time_labels = f_s[letter]['correct_index'][:]
                dt_index = np.unique(pd.to_datetime(time_labels).date)

                log_tau_ts_quadratic = f_q[letter][group_label][:,2]
                
                beta_free_plot, = ax2.plot(dt_index, np.exp(log_tau_ts), c = 'black', label = '\N{greek small letter tau}, \N{greek small letter beta} free')
                beta_2_plot, = ax2.plot(dt_index, np.exp(log_tau_ts_quadratic), c = 'black', linestyle = '--', label = '\N{greek small letter tau}, \N{greek small letter beta} =2')
                ax2.set_ylabel('\N{greek small letter tau}')
                #ax2.locator_params(axis = 'x')
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b. %y'))
                
                ax2.vlines([starts, ends], min(np.exp(log_tau_ts_quadratic)), max(np.exp(log_tau_ts_quadratic)), v_line_colors)

                for i in range(len(regime_dict['starts indeces'])):
                    ax2.fill_betweenx([min(np.exp(log_tau_ts_quadratic)), max(np.exp(log_tau_ts_quadratic))], dt_index[start_inds[i]], dt_index[end_inds[i]], color = v_line_colors[i], alpha = 0.2)

                ax_2_b = ax2.twinx()
                clpbf_plot, = ax_2_b.plot(dt_index, np.nancumsum(f_s[letter]['log_evidence'][:]-f_q[letter]['log_evidence'][:]), label = 'CLPBF', linestyle =':', color = 'black')
                ax_2_b.set_ylabel(r'BAYES FACTOR')

                handles = [beta_free_plot, beta_2_plot, clpbf_plot]
                labels = [h.get_label() for h in handles]
                ax2.legend(handles, labels)
                ax_2_b.spines["right"].set_visible(True)
                #ax2.set_title('OBSERVATION UNCERTAINTY')
                ax2.set_xlabel('DATE')
                #ax2.sharex(axs[0])

                # ------ 3rd plot -----------

                mask = np.ones(len(f_s[letter]['mean']), dtype=bool)

                for i in range(len(regime_dict['starts indeces'])):
                    mask[regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]] = False

                mean_raw = f_s[letter]['mean']
                var_raw = f_s[letter]['variance']
                
                ax3.set_ylim(min(var_raw), max(var_raw))

                mean = mean_raw[mask]
                var = var_raw[mask]

                ax3.scatter(mean, var, c= 'black', s = 1, label = 'days not in regimes')

                #ax3.set_title(r'$DAILY  m$ - $s^2$ RELATION')

                markers = ['x', '<', 'o', '>', '^', 'v', '3']

                for i in range(len(regime_dict['starts indeces'])):

                    mean = f_s[letter]['mean'][regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]
                    var = f_s[letter]['variance'][regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]

                    if regime_padding ==0:
                        upper_regime_padding = -len(mean)
                    
                    x_array = np.linspace(min(mean), max(mean), 50)

                    if plot_ribbon:
                        plot_indeces = np.arange(start = regime_dict['starts indeces'][i], stop = regime_dict['ends indeces'][i])

                        for j in plot_indeces:
                            
                            plot_alpha = log_alpha_ts[j]
                            plot_beta = beta_ts[j]
                            y_array = np.exp(plot_alpha)*x_array**plot_beta
                            ax3.plot( x_array, y_array, color = 'grey', alpha = 0.2)

                    plot_mid = regime_dict['plot_mids'][i]

                    plot_alpha = log_alpha_ts[plot_mid]
                    plot_beta = beta_ts[plot_mid]
                    y_array = np.exp(plot_alpha)*x_array**plot_beta
                    ax3.plot( x_array, y_array, label = fr'$s^2 = {np.exp(plot_alpha):.2e}\cdot m^{{{plot_beta:.2f}}}$', c= colors[i])

                    sc = ax3.scatter(mean[regime_padding:-upper_regime_padding], var[regime_padding:-upper_regime_padding], s=15, label = f"{regime_dict['starts'][i].strftime('%d %b. %y')} - {regime_dict['ends'][i].strftime('%d %b. %y')}", marker = markers[i]) # c= colors
                                        

                    offset_alpha = np.exp(log_tau_ts)/(np.cos(np.arctan(beta_ts)))
                    offset_x = np.exp(log_tau_ts[plot_mid])/(np.sin(np.arctan(beta_ts[plot_mid])))

                    x_upper_min = np.exp(np.log(min(mean))-offset_x)
                    x_upper_max = np.exp(np.log(max(mean))-offset_x)

                    x_lower_min = np.exp(np.log(min(mean))+offset_x)
                    x_lower_max = np.exp(np.log(max(mean))+offset_x)

                    x_array_upper_ci = np.linspace(x_upper_min, x_upper_max, 50)
                    x_array_lower_ci = np.linspace(x_lower_min, x_lower_max, 50)

                    ax3.plot(x_array_lower_ci, np.exp(log_alpha_ts[plot_mid]-ci_tau_multiplier*offset_alpha[plot_mid])*x_array_lower_ci**beta_ts[plot_mid], linestyle = ':', c= colors[i])
                    ax3.plot(x_array_upper_ci, np.exp(log_alpha_ts[plot_mid]+ci_tau_multiplier*offset_alpha[plot_mid])*x_array_upper_ci**beta_ts[plot_mid], linestyle = ':', c= colors[i])#, label = '2 \N{greek small letter tau} prediction interval')

                    '''square TL comparison model'''
                    beta_ts_quad = f_q[letter][group_label][:,1]
                    
                    log_alpha_ts_quad = f_q[letter][group_label][:,0]

                    y_array = np.exp(log_alpha_ts_quad[plot_mid])*x_array**beta_ts_quad[plot_mid]
                    ax3.plot( x_array, y_array, label = fr'$s^2 = {np.exp(log_alpha_ts_quad[plot_mid]):.2e}\cdot m^{{{beta_ts_quad[plot_mid]:.2g}}}$', linestyle = '--', c= colors[i])

                ax3.set_ylabel(r'$s^2$ in $L^2/s^2$')
                ax3.set_xlabel(r'$m$ in $L/s$')

                x_plot_min_max_diff = max(mean_raw)- min(mean_raw)
                lower_x_padding = 0.05*x_plot_min_max_diff
                upper_x_padding = 0.5*x_plot_min_max_diff

                ax3.set_xlim(min(mean_raw)-lower_x_padding, max(mean_raw)+upper_x_padding)

                if logarithmic_scatter:
                    ax3.set_xscale('log')
                    ax3.set_yscale('log')
                    
                ax3.legend()
                #plt.tight_layout()
                plt.setp(ax1.get_xticklabels(), visible=False)
                fig.set_constrained_layout_pads(hspace=0.02)

                if file_type == 'png':
                    plt.savefig(f'{file_prefix}{letter}.png', bbox_inches='tight')
                elif file_type == 'svg':
                    plt.savefig(f'{file_prefix}{letter}.svg', bbox_inches='tight', transparent=True)
                plt.close(fig)

def plot_wsnd_no_regimes(free_beta_result_path,
                        quadratic_result_path,
                        export_file_path,
                        file_type,
                        summary_statistic,
                        credible_interval,
                        ci_tau_multiplier,
                        logarithmic_scatter,
                        plot_beta_lim,
                        plot_ribbon):
    '''This generates the full result plots for the WSND data presented in the paper.
    As the reported time series is to short to confidently separate multiple regimes, no regime separation is performed.'''

    # Colors from the currently active style
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    letters = ['WSND']

    upper_ylim = plot_beta_lim

    for letter in letters:
        with h5py.File(free_beta_result_path, "r") as f_s:

            regime_dict = {}

            fig, axs = plt.subplots(3, 1, figsize=(10, 12))

            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]
            
            mean = f_s[letter]['mean'][:]
            var = f_s[letter]['variance'][:]

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

            match credible_interval:
                case '90 %':
                    log_alpha_lower_ci = f_s[letter]['5%'][:,0]
                    log_alpha_upper_ci = f_s[letter]['95%'][:,0]

                    beta_lower_ci = f_s[letter]['5%'][:,1]
                    beta_upper_ci = f_s[letter]['95%'][:,1]

                    log_tau_lower_ci = f_s[letter]['5%'][:,2]
                    log_tau_upper_ci = f_s[letter]['95%'][:,2]

                case '50 %':
                    log_alpha_lower_ci = f_s[letter]['25%'][:,0]
                    log_alpha_upper_ci = f_s[letter]['75%'][:,0]

                    beta_lower_ci = f_s[letter]['25%'][:,1]
                    beta_upper_ci = f_s[letter]['75%'][:,1]

                    log_tau_lower_ci = f_s[letter]['25%'][:,2]
                    log_tau_upper_ci = f_s[letter]['75%'][:,2]

            time_labels = f_s[letter]['correct_index'][:]
            dt_index = np.unique(pd.to_datetime(time_labels).date)

            regime_dict['starts'] = dt_index[0]
            regime_dict['ends'] = dt_index[-1]

            regime_dict['starts indeces'] = [0]
            regime_dict['ends indeces'] = [len(dt_index)]

            beta_inds = np.argsort(beta_ts)
            median_beta_ind = len(dt_index)-2#beta_inds[int(0.5*len(dt_index))+1]
            regime_dict['plot_mids'] = [median_beta_ind]

            # ---------- 1st Plot -----------------------------
            '''Depicts the time evolution of the estimated beta and log(alpha) for the unconstrained beta estimation'''

            _beta_plot, = ax1.plot(dt_index, beta_ts, c = 'black', label = r'$\beta$')
            ci_b_plot, = ax1.plot(dt_index, beta_lower_ci, c = 'black', linestyle =':', label = fr'{credible_interval} CI $\beta$')
            ax1.plot(dt_index, beta_upper_ci, c = 'black', linestyle =':')
            ax1.set_ylim(0, upper_ylim)
            ax1.set_ylabel(r'$\beta$')
            
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%b.'))

            ax_1_b = ax1.twinx()
            log_alpha_plot, = ax_1_b.plot(dt_index, log_alpha_ts, c = 'grey', linestyle = '--', label = r'$\mathrm{log(}\alpha\mathrm{)}$')
            ax_1_b.plot(dt_index, log_alpha_lower_ci, c = 'grey', linestyle = ':')
            ci_a_plot, = ax_1_b.plot(dt_index, log_alpha_upper_ci, c = 'grey', linestyle = ':', label = fr'{credible_interval} CI $\alpha$')
            ax_1_b.set_ylim(0, upper_ylim*(np.mean(log_alpha_ts)/np.mean(beta_ts)))
            ax_1_b.set_ylabel(r'$\mathrm{log(} \alpha \mathrm{)}$')

            handles = [_beta_plot, ci_b_plot, log_alpha_plot, ci_a_plot]
            labels = [h.get_label() for h in handles]

            ax_1_b.spines["right"].set_visible(True)
            ax1.legend(handles, labels)
            ax1.tick_params(labelbottom=False)


            # ---------- 2nd Plot -----------------------------
            '''Depicts the time series for the scale variables tau and the cumulative predictive log Bayes Factor for both models'''
            with h5py.File(quadratic_result_path, "r") as f_q:
                
                time_labels = f_s[letter]['correct_index'][:]
                dt_index = np.unique(pd.to_datetime(time_labels).date)

                log_tau_ts_quadratic = f_q[letter][group_label][:,2]
                
                beta_free_plot, = ax2.plot(dt_index, np.exp(log_tau_ts), c = 'black', label = '\N{greek small letter tau}, \N{greek small letter beta} free')
                beta_2_plot, = ax2.plot(dt_index, np.exp(log_tau_ts_quadratic), c = 'black', linestyle = '--', label = '\N{greek small letter tau}, \N{greek small letter beta} =2')
                ax2.set_ylabel('\N{greek small letter tau}')
                #ax2.locator_params(axis = 'x')
                ax2.xaxis.set_major_locator(mdates.DayLocator(interval=8))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%b.'))
                
                ax_2_b = ax2.twinx()
                clpbf_plot, = ax_2_b.plot(dt_index, np.nancumsum(f_s[letter]['log_evidence'][:]-f_q[letter]['log_evidence'][:]), label = 'CLPBF', linestyle =':', color = 'black')
                ax_2_b.set_ylabel(r'BAYES FACTOR')

                handles = [beta_free_plot, beta_2_plot, clpbf_plot]
                labels = [h.get_label() for h in handles]
                ax2.legend(handles, labels)
                ax_2_b.spines["right"].set_visible(True)
                ax2.set_xlabel('DATE')

                # -------------- 3rd plot ------------------------
                '''Depicts the scatter plots of the data and both estimated TL relationships'''

                mean = f_s[letter]['mean']
                var = f_s[letter]['variance']
                
                ax3.set_ylim(0.95*min(var), 1.05*max(var))

                ax3.scatter(mean, var, c= 'black', s = 15, label = 'all days', marker = 'x')

                for i in range(len(regime_dict['starts indeces'])):

                    mean = f_s[letter]['mean'][regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]
                    var = f_s[letter]['variance'][regime_dict['starts indeces'][i]:regime_dict['ends indeces'][i]]
                    
                    x_array = np.linspace(min(mean), max(mean), 50)

                    if plot_ribbon:
                        plot_indeces = np.arange(start = regime_dict['starts indeces'][i], stop = regime_dict['ends indeces'][i])

                        for j in plot_indeces:
                            
                            plot_alpha = log_alpha_ts[j]
                            plot_beta = beta_ts[j]
                            y_array = np.exp(plot_alpha)*x_array**plot_beta
                            ax3.plot( x_array, y_array, color = 'grey', alpha = 0.2)

                    plot_mid = regime_dict['plot_mids'][i]

                    plot_alpha = log_alpha_ts[plot_mid]
                    plot_beta = beta_ts[plot_mid]
                    y_array = np.exp(plot_alpha)*x_array**plot_beta
                    ax3.plot( x_array, y_array, label = fr'$s^2 = {np.exp(plot_alpha):.2e}\cdot m^{{{plot_beta:.2f}}}$', c='black')

                    offset_alpha = np.exp(log_tau_ts)/(np.cos(np.arctan(beta_ts)))
                    offset_x = np.exp(log_tau_ts[plot_mid])/(np.sin(np.arctan(beta_ts[plot_mid])))

                    x_upper_min = np.exp(np.log(min(mean))-offset_x)
                    x_upper_max = np.exp(np.log(max(mean))-offset_x)

                    x_lower_min = np.exp(np.log(min(mean))+offset_x)
                    x_lower_max = np.exp(np.log(max(mean))+offset_x)

                    x_array_upper_ci = np.linspace(x_upper_min, x_upper_max, 50)
                    x_array_lower_ci = np.linspace(x_lower_min, x_lower_max, 50)

                    ax3.plot(x_array_lower_ci, np.exp(log_alpha_ts[plot_mid]-ci_tau_multiplier*offset_alpha[plot_mid])*x_array_lower_ci**beta_ts[plot_mid], linestyle = ':', c='black')
                    ax3.plot(x_array_upper_ci, np.exp(log_alpha_ts[plot_mid]+ci_tau_multiplier*offset_alpha[plot_mid])*x_array_upper_ci**beta_ts[plot_mid], linestyle = ':', c='black', label = '2 \N{greek small letter tau} prediction interval')

                    '''square TL comparison model'''
                    beta_ts_quad = f_q[letter][group_label][:,1]
                    
                    log_alpha_ts_quad = f_q[letter][group_label][:,0]

                    y_array = np.exp(log_alpha_ts_quad[plot_mid])*x_array**beta_ts_quad[plot_mid]
                    ax3.plot( x_array, y_array, label = fr'$s^2 = {np.exp(log_alpha_ts_quad[plot_mid]):.2e}\cdot m^{{{beta_ts_quad[plot_mid]:.2g}}}$', linestyle = '--', c='black')

                ax3.set_ylabel(r'$s^2$ in $L^2/s^2$')
                ax3.set_xlabel(r'$m$ in $L/s$')

                x_plot_min_max_diff = max(mean)- min(mean)
                lower_x_padding = 0.05*x_plot_min_max_diff
                upper_x_padding = 0.5*x_plot_min_max_diff

                ax3.set_xlim(min(mean)-lower_x_padding, max(mean)+upper_x_padding)

                if logarithmic_scatter:
                    ax3.set_xscale('log')
                    ax3.set_yscale('log')
                    
                ax3.legend()
                plt.setp(ax1.get_xticklabels(), visible=False)
                fig.set_constrained_layout_pads(hspace=0.02)
                
                '''Save plots to files'''
                if file_type == 'png':
                    plt.savefig(f'{export_file_path}.png', bbox_inches='tight')
                elif file_type == 'svg':
                    plt.savefig(f'{export_file_path}.svg', bbox_inches='tight', transparent=True)
                plt.close(fig)

def main_plotter(
    file_type_text,
    file_type_ga,
    summary_statistic,
    credible_interval,
    ci_tau_multiplier,
    regime_quantile_threshold,
    regime_identification_parameter,
    logarithmic_scatter,
    plot_beta_lim,
    plot_ribbon,
    regime_padding,

    free_beta_result_path_longest_regimes,
    quadratic_result_path_longest_regimes,
    no_displayed_regimes_longest_regimes,
    file_prefix_longest_regimes,

    free_beta_result_path_dma,
    quadratic_result_path_dma,
    file_prefix_dma,
    no_displayed_regimes_dma,

    free_beta_result_path_wsnd,
    quadratic_result_path_wsnd,
    export_file_path_wsnd):

    '''Generate plots for graphical abstract'''

    # representative day with marked mean demand
    single_representative_day(file_type = file_type_ga)

    # scatter and estimates only for thw two longest regimes
    plot_dmas_vertical_only_beta(free_beta_result_path = free_beta_result_path_longest_regimes,
                        quadratic_result_path = quadratic_result_path_longest_regimes,
                        file_type = file_type_ga,
                        summary_statistic = summary_statistic,
                        regime_quantile_threshold = regime_quantile_threshold,
                        regime_identification_parameter = regime_identification_parameter,
                        no_displayed_regimes = no_displayed_regimes_longest_regimes,
                        logarithmic_scatter = logarithmic_scatter,
                        plot_beta_lim =plot_beta_lim,
                        plot_ribbon = plot_ribbon,
                        regime_padding = regime_padding,
                        file_prefix = file_prefix_longest_regimes)

    
    '''Generate plots for main text'''

    # representative week

    plot_representative_input_data(file_type = file_type_text)

    # BWDF results

    plot_dmas_vertical(free_beta_result_path = free_beta_result_path_dma,
                        quadratic_result_path = quadratic_result_path_dma,
                        file_prefix = file_prefix_dma,
                        file_type = file_type_text,
                        summary_statistic = summary_statistic,
                        credible_interval = credible_interval,
                        ci_tau_multiplier = ci_tau_multiplier,
                        regime_quantile_threshold = regime_quantile_threshold,
                        regime_identification_parameter = regime_identification_parameter,
                        no_displayed_regimes = no_displayed_regimes_dma,
                        logarithmic_scatter = logarithmic_scatter,
                        plot_beta_lim =plot_beta_lim,
                        plot_ribbon = plot_ribbon,
                        regime_padding = regime_padding)

    # WSND results
    plot_wsnd_no_regimes(free_beta_result_path = free_beta_result_path_wsnd,
                        quadratic_result_path = quadratic_result_path_wsnd,
                        export_file_path = export_file_path_wsnd,
                        file_type = file_type_text,
                        summary_statistic = summary_statistic,
                        credible_interval = credible_interval,
                        ci_tau_multiplier = ci_tau_multiplier,
                        logarithmic_scatter = logarithmic_scatter,
                        plot_beta_lim = plot_beta_lim,
                        plot_ribbon = plot_ribbon)

    return 'Plotting done.'

if __name__ == "__main__":
    main_plotter(
    file_type_text = 'png',
    file_type_ga = 'svg',
    summary_statistic = 'median',
    credible_interval = '50 %',
    ci_tau_multiplier = 2,
    regime_quantile_threshold = 0.8,
    regime_identification_parameter = 'derivative_stability',
    logarithmic_scatter = True,
    plot_beta_lim =25,
    plot_ribbon = False,
    regime_padding = 0,

    free_beta_result_path_longest_regimes = 'particle_filter_results_standard_BWDF.h5',
    quadratic_result_path_longest_regimes = 'particle_filter_results_quadratic_BWDF.h5',
    no_displayed_regimes_longest_regimes = 2,
    file_prefix_longest_regimes = 'two_regimes_for_abstract',

    free_beta_result_path_dma = 'particle_filter_results_standard_BWDF.h5',
    quadratic_result_path_dma = 'particle_filter_results_quadratic_BWDF.h5',
    file_prefix_dma = 'results_dma_',
    no_displayed_regimes_dma = 3,

    free_beta_result_path_wsnd = 'particle_filter_results_WSND.h5',
    quadratic_result_path_wsnd = 'particle_filter_results_WSND_quadratic.h5',
    export_file_path_wsnd = 'results_wsnd_'
    )
    