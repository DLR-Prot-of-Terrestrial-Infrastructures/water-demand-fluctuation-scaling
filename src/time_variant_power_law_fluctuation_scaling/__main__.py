# Copyright (c) YYYY German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Bernhard Jonathan Sattler, Bernhard.sattler@dlr.de
# Date of creation: 2026-10-09

import pandas as pd
import numpy as np
import h5py

from PF_type_estimate import estimate_taylors_law
from data_cleaning import import_data, clean_data, clean_wsnd_data
from results_analysis import combine_posteriors, combine_log_evidences, reduce_to_marginal_credible_intervals
from plotters import main_plotter
from generate_table_2 import generate_csv, clean_csv

from concurrent.futures import ProcessPoolExecutor

# for reproducibility of the plots, we use this random seed in reference to the year of Taylors seminal publication.
# Different seeds can lead to slight variations in the obtained regime borders due to Monte-Carlo variation.
SEED = 1961

def analyze_column(letter, series, mc_runs,
                   ess_threshold, proportion_of_prior,
                   beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count, rng):
    '''runs the analysis for a column of the cleaned input DataFrame of the BWDF data.
    letter:
    series: data as a pd.Series
    mc_runs: number of Monte Carlo runs performed in this batch run.
    ess_threshold: Threshold of the effective sample size (ESS = sum(1/w^2)) at which resampling is performed.
    proportion_of_prior: proportion of the prior that is inserted into the resampled posterior.
    beta_min, beta_max: range of beta for the prior.
    log_sigma_min, log_sigma_max: range of the observation uncertainty log_sigma (or log(tau) in the paper for the prior.)
    particle_count: number N of the particles of the particle filter.
    rng: local seeded random number generator.'''
    
    result_dict ={}

    data_array = series.values

    result_dict['raw_data'] = data_array.astype(np.float32)
    result_dict['correct_index'] = series[~series.isna()].index.view('int64')

    mean = series.resample('d').mean().dropna().astype(np.float64).values
    result_dict['mean'] = mean
    var = series.resample('d').var().dropna().astype(np.float64).values
    result_dict['variance'] = var

    run_rngs = rng.spawn(mc_runs)
    for i in range(mc_runs):
        print('MC run ' + str(i) + ' of ' + str(mc_runs-1) + ' for DMA ' + letter)
        analysis_results_array = estimate_taylors_law(mean, var, ess_threshold = ess_threshold, proportion_of_prior = proportion_of_prior,
                                                     beta_min = beta_min, beta_max = beta_max, log_sigma_min = log_sigma_min, log_sigma_max = log_sigma_max,
                                                    particle_count = particle_count, rng=run_rngs[i])
        
        result_dict[f'mc_run_{i}'] = analysis_results_array

    return result_dict


def result_logger(result_dict, filename, mc_runs, ess_threshold, proportion_of_prior, beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count):
    '''Logs the analysis results.

    result_dict: results of the analyze_column method.
    filename: docstring of the output file.
    mc_runs: number of Monte Carlo runs performed in this batch run.
    ess_threshold: Threshold of the effective sample size (ESS = sum(1/w^2)) at which resampling is performed.
    proportion_of_prior: proportion of the prior that is inserted into the resampled posterior.
    beta_min, beta_max: range of beta for the prior.
    log_sigma_min, log_sigma_max: range of the observation uncertainty log_sigma (or log(tau) in the paper for the prior.)
    particle_count: number N of the particles of the particle filter.'''

    with h5py.File(filename, "a") as f:

        # Initialize input group only if it doesn't exist
        if 'input' not in f:

            group = f.create_group('input')
            group.attrs.create('ess_threshold', ess_threshold)
            group.attrs.create('proportion_of_prior', proportion_of_prior)
            group.attrs.create('beta_min', beta_min)
            group.attrs.create('beta_max', beta_max)
            group.attrs.create('log_sigma_min', log_sigma_min)
            group.attrs.create('log_sigma_max', log_sigma_max)
            group.attrs.create('particle_count', particle_count)

            group.create_dataset("raw_data", data = result_dict['raw_data'])
            group.create_dataset("correct_index", data = result_dict['correct_index'])

            group.create_dataset("mean", data = result_dict['mean'])
            group.create_dataset("variance", data = result_dict['variance'])
                
        # Find existing MC groups
        mc_groups = []
        
        # Extract numbers from group names
        for group_name in f.keys():
            try:
                num = int(group_name.split('_')[1])
                mc_groups.append(num)
            except (ValueError, IndexError):
                continue

        # Determine the next available number
        next_num = max(mc_groups) + 1 if mc_groups else 0

        for i in range(mc_runs):

            mc_idx = next_num + i
            
            # Create forward group
            group_name = f'mcgroup_{mc_idx}'
            if group_name not in f:

                group = f.create_group(group_name)

                group.create_dataset("log_evidence", data = result_dict[f'mc_run_{i}'][0])
                group.create_dataset("posterior_map", data = result_dict[f'mc_run_{i}'][1]) # in theory, this does not need the parameters and carries unnecessary data -> fix later
                group.create_dataset("parameter_space_map", data = result_dict[f'mc_run_{i}'][2])

        f.close()

def run_analysis(ess_threshold, proportion_of_prior,
                 beta_max, beta_min, log_sigma_min, log_sigma_max, particle_count,
                 alpha_level, alpha_level_tighter,
                 mc_runs, batches, import_excel, 
                 intermediate_h5_appendix, output_h5, all, rng):
    '''runs the import, preprocessing, and analysis for the input Excel of the BWDF data.

    mc_runs: number of Monte Carlo runs performed in each batch run.
    batches: number of batches.
    ess_threshold: Threshold of the effective sample size (ESS = sum(1/w^2)) at which resampling is performed.
    proportion_of_prior: proportion of the prior that is inserted into the resampled posterior.
    beta_min, beta_max: range of beta for the prior.
    log_sigma_min, log_sigma_max: range of the observation uncertainty log_sigma (or log(tau) in the paper for the prior.)
    particle_count: number N of the particles of the particle filter.
    alpha_level: reported percentiles outside the wider credible interval (not reported in the paper, default is 10% outside of interval -> 5% and 95% CI)
    alpha_level_tighter: reported percentiles outside the tighter credible interval (reported in the paper, default is 50% outside of interval -> 25% and 75% CI)
    import_excel: docstring to the import file (excel) of the BWDF data.
    intermediate_h5_appendix: end of the docstring of the h5 file, that will be used.
    output_h5: docstring for the .h5 file saving the analysis output.
    all: True, if all data rows should be analyzed. This can be adjusted for debugging according to import_data.
    rng: seeded random number generator to enable reproducability.'''
    
    #this typically has to be set to True, but if the second step failed one can set it to False to just do the aggregation of filtered results.
    initial_analysis = True

    # imports the data and sets index to DateTime format
    import_df = import_data(import_excel = import_excel, all=all)

    # deletes any incomplete days and day light savings time changes
    cleaned_full_timeline_no_daylight_saving_df = clean_data(import_df)

    # seperate arrays and letters to reference for the analysis
    seperated_timelines = [cleaned_full_timeline_no_daylight_saving_df[col] for col in cleaned_full_timeline_no_daylight_saving_df.columns]
    column_letters = [str(col)[4] for col in cleaned_full_timeline_no_daylight_saving_df.columns]


    if initial_analysis:

        # creating loccal rngs for each stream
        number_of_timelines = len(column_letters)
        local_rngs = rng.spawn(batches*number_of_timelines)

        for batch in range(batches):
            # run the analysis on all columns of the Data Set
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(analyze_column, column_letters[i], seperated_timelines[i],
                                    mc_runs, ess_threshold, proportion_of_prior,
                                    beta_min, beta_max, log_sigma_min, log_sigma_max,
                                    particle_count, local_rngs[i+batch*number_of_timelines])
                    for i in range(number_of_timelines)
                ]

            for i, future in enumerate(futures):
                filename = column_letters[i] + intermediate_h5_appendix
                result_logger(future.result(), filename, mc_runs, ess_threshold, proportion_of_prior, beta_min, beta_max, log_sigma_min, log_sigma_max,
                              particle_count)

    reduce = True

    if reduce:
        combined_dict = combine_posteriors(column_letters, doc_string = intermediate_h5_appendix, doc_count=1) # column_letters become keys for the combined dict -> also for median_dict
        log_evidences = combine_log_evidences(column_letters, doc_string = intermediate_h5_appendix, doc_count=1)

        # this ection is depreciated, since its too computationally expensive
        # median_dict = reduce_to_geometric_median_particle(combined_dict=combined_dict)

        credible_interval_dict  = reduce_to_marginal_credible_intervals(combined_dict=combined_dict, alpha_level = alpha_level, alpha_level_tighter = alpha_level_tighter)

        with h5py.File(output_h5, "a") as f:

            for i, key in enumerate(credible_interval_dict.keys()):

                group = f.create_group(key)
                group.attrs.create('ess_threshold', ess_threshold)
                group.attrs.create('proportion_of_prior', proportion_of_prior)
                group.attrs.create('beta_min', beta_min)
                group.attrs.create('beta_max', beta_max)
                group.attrs.create('log_sigma_min', log_sigma_min)
                group.attrs.create('log_sigma_max', log_sigma_max)
                group.attrs.create('particle_count', particle_count)
                group.attrs.create('alpha_level', alpha_level)
                group.attrs.create('alpha_level_tighter', alpha_level_tighter)
                group.attrs.create('total MC runs', mc_runs*batches)

                filename = key + intermediate_h5_appendix

                with h5py.File(filename, "r") as f_input:

                    raw_data = f_input['input']['raw_data'][:]
                    correct_index = f_input['input']['correct_index'][:]

                    _mean = f_input['input']['mean'][:]
                    _variance = f_input['input']['variance'][:]

                group.create_dataset("raw_data", data = raw_data)
                group.create_dataset("correct_index", data = correct_index)

                group.create_dataset("mean", data = _mean)
                group.create_dataset("variance", data = _variance)

                group.create_dataset("median", data = credible_interval_dict[key]["median"])
                group.create_dataset("mean posterior", data = credible_interval_dict[key]["mean"])
                group.create_dataset("MAP", data = credible_interval_dict[key]["maximum posterior"])
                group.create_dataset("5%", data = credible_interval_dict[key]["lower credible interval"])
                group.create_dataset("95%", data = credible_interval_dict[key]["upper credible interval"])

                group.create_dataset("25%", data = credible_interval_dict[key]["lower credible interval tight ci"])
                group.create_dataset("75%", data = credible_interval_dict[key]["upper credible interval tight ci"])

                group.create_dataset("log_evidence", data = log_evidences[key][:])

    termination_string = f'Analysis on the basis {intermediate_h5_appendix} sample values done.'

    return termination_string

def run_wsnd_analysis(
    ess_threshold,
    proportion_of_prior,
    beta_min,
    beta_max,
    log_sigma_min,
    log_sigma_max,
    particle_count,
    alpha_level,
    alpha_level_tighter,
    mc_runs,
    batches,
    import_string,
    intermediate_h5_appendix,
    output_h5,
    rng
    ):
    '''Method analogus to the 'run_analysis' method but for the WSND data set.
    
    mc_runs: number of Monte Carlo runs performed in each batch run.
    batches: number of batches.
    ess_threshold: Threshold of the effective sample size (ESS = sum(1/w^2)) at which resampling is performed.
    proportion_of_prior: proportion of the prior that is inserted into the resampled posterior.
    beta_min, beta_max: range of beta for the prior.
    log_sigma_min, log_sigma_max: range of the observation uncertainty log_sigma (or log(tau) in the paper for the prior.)
    particle_count: number N of the particles of the particle filter.
    alpha_level: reported percentiles outside the wider credible interval (not reported in the paper, default is 10% outside of interval -> 5% and 95% CI)
    alpha_level_tighter: reported percentiles outside the tighter credible interval (reported in the paper, default is 50% outside of interval -> 25% and 75% CI)
    import_string: docstring to the import file (csv) of the WSND inflow data.
    intermediate_h5_appendix: end of the docstring of the h5 file, that will be used.
    output_h5: docstring for the .h5 file saving the analysis output.
    rng: seeded random number generator to enable reproducability.'''
    
    # data import
    raw_df=pd.read_csv(import_string, index_col='Timestamp', date_format = "%m/%d/%Y %H:%M")

    # Data cleaning
    hourly_avg = clean_wsnd_data(raw_df)

    # convert to L/s
    hourly_avg["Flow (L/s)"] = hourly_avg["Flow (m3/hr)"]/3.6

    # creating loccal rngs for each stream
    local_rngs = rng.spawn(batches)

    for batch in range(batches):
        # run the analysis on the Data Set
        result = analyze_column('WSND', hourly_avg["Flow (L/s)"],
                                mc_runs, ess_threshold, proportion_of_prior,
                                beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count, local_rngs[batch])

        filename = 'WSND' + intermediate_h5_appendix

        #logging results
        result_logger(result, filename, mc_runs, ess_threshold, proportion_of_prior, beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count)

    combined_dict = combine_posteriors(['WSND'], doc_string = intermediate_h5_appendix, doc_count=1) # column_letters become keys for the combined dict -> also for median_dict
    log_evidences = combine_log_evidences(['WSND'], doc_string = intermediate_h5_appendix, doc_count=1)

    credible_interval_dict = reduce_to_marginal_credible_intervals(combined_dict=combined_dict, alpha_level = alpha_level, alpha_level_tighter = alpha_level_tighter)

    with h5py.File(output_h5, "a") as f:

        for i, key in enumerate(credible_interval_dict.keys()):

            group = f.create_group(key)
            group.attrs.create('ess_threshold', ess_threshold)
            group.attrs.create('proportion_of_prior', proportion_of_prior)
            group.attrs.create('beta_min', beta_min)
            group.attrs.create('beta_max', beta_max)
            group.attrs.create('log_sigma_min', log_sigma_min)
            group.attrs.create('log_sigma_max', log_sigma_max)
            group.attrs.create('particle_count', particle_count)
            group.attrs.create('alpha_level', alpha_level)
            group.attrs.create('alpha_level_tighter', alpha_level_tighter)
            group.attrs.create('total MC runs', mc_runs*batches)

            filename = key + intermediate_h5_appendix

            with h5py.File(filename, "r") as f_input:

                raw_data = f_input['input']['raw_data'][:]
                correct_index = f_input['input']['correct_index'][:]

                _mean = f_input['input']['mean'][:]
                _variance = f_input['input']['variance'][:]

            group.create_dataset("raw_data", data = raw_data)
            group.create_dataset("correct_index", data = correct_index)

            group.create_dataset("mean", data = _mean)
            group.create_dataset("variance", data = _variance)

            group.create_dataset("median", data = credible_interval_dict[key]["median"])
            group.create_dataset("mean posterior", data = credible_interval_dict[key]["mean"])
            group.create_dataset("MAP", data = credible_interval_dict[key]["maximum posterior"])
            group.create_dataset("5%", data = credible_interval_dict[key]["lower credible interval"])
            group.create_dataset("95%", data = credible_interval_dict[key]["upper credible interval"])

            group.create_dataset("25%", data = credible_interval_dict[key]["lower credible interval tight ci"])
            group.create_dataset("75%", data = credible_interval_dict[key]["upper credible interval tight ci"])

            group.create_dataset("log_evidence", data = log_evidences[key][:])

    return f'Analysis on the basis {intermediate_h5_appendix} sample values done.'

if __name__ == "__main__":

    common_ess_threshold = 0.2
    common_proportion_of_prior = 0.001
    common_log_sigma_min= -6
    common_log_sigma_max= -1
    common_particle_count = 10000
    common_alpha_level = 0.1
    common_alpha_level_tighter = 0.5
    common_mc_runs = 2
    common_batches = 5
    
    # We need one seed for each Monte Carlo run in each analysis stream
    parent_rng = np.random.default_rng(SEED)
    # Now we generate 4 child seeds; one for each subsequent run_analysis - type method
    child_rngs = parent_rng.spawn(4)

    return_string = run_analysis(ess_threshold = common_ess_threshold,
                proportion_of_prior = common_proportion_of_prior,
                beta_min = 0,
                beta_max = 50,
                log_sigma_min= common_log_sigma_min,
                log_sigma_max= common_log_sigma_max,
                particle_count = common_particle_count,
                alpha_level = common_alpha_level,
                alpha_level_tighter = common_alpha_level_tighter,
                mc_runs = common_mc_runs,
                batches = common_batches,
                import_excel = "Inflow_Data_4.xlsx", 
                intermediate_h5_appendix = "_intermediate_standard_BWDF.h5",
                output_h5 = "particle_filter_results_standard_BWDF.h5",
                all= True,
                rng = child_rngs[0])
    
    print(return_string)

    return_string = run_analysis(ess_threshold = common_ess_threshold,
                proportion_of_prior = common_proportion_of_prior,
                beta_min = 2,
                beta_max = 2,
                log_sigma_min= common_log_sigma_min,
                log_sigma_max= common_log_sigma_max,
                particle_count = common_particle_count,
                alpha_level = common_alpha_level,
                alpha_level_tighter = common_alpha_level_tighter,
                mc_runs = common_mc_runs,
                batches = common_batches,
                import_excel = "Inflow_Data_4.xlsx", 
                intermediate_h5_appendix = "_intermediate_quadratic_BWDF.h5",
                output_h5 = "particle_filter_results_quadratic_BWDF.h5",
                all= True,
                rng = child_rngs[1])

    print(return_string)

    #run analysis with \beta = free
    return_string = run_wsnd_analysis(
                ess_threshold = common_ess_threshold,
                proportion_of_prior = common_proportion_of_prior,
                beta_min = 0,
                beta_max = 50,
                log_sigma_min= common_log_sigma_min,
                log_sigma_max= common_log_sigma_max,
                particle_count = common_particle_count,
                alpha_level = common_alpha_level,
                alpha_level_tighter = common_alpha_level_tighter,
                mc_runs = common_mc_runs,
                batches = common_batches,
                import_string = "inflow.csv",
                intermediate_h5_appendix = "_intermediate_WSND.h5",
                output_h5 = "particle_filter_results_WSND.h5",
                rng = child_rngs[2]
                )

    print(return_string)

    #run analysis with \beta = 2
    return_string_quadratic = run_wsnd_analysis(
                ess_threshold = common_ess_threshold,
                proportion_of_prior = common_proportion_of_prior,
                beta_min = 2,
                beta_max = 2,
                log_sigma_min= common_log_sigma_min,
                log_sigma_max= common_log_sigma_max,
                particle_count = common_particle_count,
                alpha_level = common_alpha_level,
                alpha_level_tighter = common_alpha_level_tighter,
                mc_runs = common_mc_runs,
                batches = common_batches,
                import_string = "inflow.csv",
                intermediate_h5_appendix = "_intermediate_WSND_quadratic.h5",
                output_h5 = "particle_filter_results_WSND_quadratic.h5",
                rng = child_rngs[3]
            )

    print(return_string_quadratic)


    plotter_return = main_plotter(
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

    print(plotter_return)

    gen_csv_return = generate_csv(free_beta_result_path_dma = 'particle_filter_results_standard_BWDF.h5',
            free_beta_result_path_wsnd = 'particle_filter_results_WSND.h5',
            export_path = 'temp_contents_table_2.csv',

            summary_statistic = 'median',
            regime_quantile_threshold = 0.8,
            regime_identification_parameter = 'derivative_stability',
            reporting_threshold = 28,)

    print(gen_csv_return)

    clean_csv_return = clean_csv(csv_path_input = 'temp_contents_table_2.csv',
                                csv_path_output = 'contents_table_2.csv')

    print(clean_csv_return)
