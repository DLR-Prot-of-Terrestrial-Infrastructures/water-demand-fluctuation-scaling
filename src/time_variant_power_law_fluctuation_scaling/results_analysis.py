# Copyright (c) YYYY German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Bernhard Jonathan Sattler, Bernhard.sattler@dlr.de
# Date of creation: 2026-10-09

import pandas as pd
import numpy as np
import h5py

def combine_posteriors(letters, doc_string, doc_count = 1):
    '''
    combines the posterior weighted particle sets obtained by the filter.

    :letters: list of letters with which the .h5 files start. Also becomes the keys for the return dict
    :doc_string: ending of the .h5 file which holds the data
    :doc_count: number of documents with the same ending that have to be combined
    '''
    
    combined_dict = {}

    for letter in letters:

        collect_parameters = []
        collect_posteriors = []

        for j in range(doc_count):
            with h5py.File(f"{letter}{doc_string}", "r") as f:

                mc_runs = int((len(list(f.keys()))-1)/2)

                for i in range(mc_runs):

                    mc_string = 'mcgroup_' + str(i)
        
                    '''import the data from the logging files'''
                    parameters = f[mc_string]['parameter_space_map'][:]
                    likelihoods = f[mc_string]['posterior_map'][:]

                    collect_parameters.append(parameters)
                    collect_posteriors.append(likelihoods)

        combined_parameters = np.concatenate(collect_parameters, axis=2)
        combined_posteriors = np.concatenate(collect_posteriors, axis=2)
        combined_posteriors /= np.sum(combined_posteriors)

        combined_dict[letter] = {
            'para' : combined_parameters,
            'weights' : combined_posteriors,
        }
    
    print('Posterior combiation done.')

    return combined_dict


def marginal_credible_intervals(parameter_space, weights, alpha_level):
    ''' Computed the marginal credible intervals for one time step.
    parameter_space: the full obtained particle set as np.Array.
    weights: the weight matrix as a np.Array.
    alpha_level: the percentag of the weight distribution outside of the marginal credible interval. E.g.: alpha_level=0.1 results in a 5% and 95% CI.'''

    one_sided_alpha = alpha_level/2

    # sort models and posterior according to the logalpha-parameter
    sort_indeces = np.argsort(parameter_space[:,:], axis=1)
    sorted_parameters = np.take_along_axis(parameter_space[:,:], sort_indeces, axis=1)
    sorted_posterior = np.take_along_axis(weights, sort_indeces, axis=1)

    # generate cumulative distribution
    cumsum_posterior = np.cumsum(sorted_posterior, axis=1)
    normalized_cumsum_posterior = cumsum_posterior/cumsum_posterior[:,-1][..., np.newaxis]

    # take the model where the cumulative distribution hits 5 %
    lower_limit_posterior_indeces = np.argmax(normalized_cumsum_posterior>=one_sided_alpha, axis=1, keepdims=True)
    lower_limit_posterior_parameters = np.take_along_axis(sorted_parameters, lower_limit_posterior_indeces, axis=1)
    
    # extract model parameters
    _parameters_lower = lower_limit_posterior_parameters[:,:]

    # take the model where the cumulative distribution hits 95 %
    upper_limit_posterior_indeces = np.argmax(normalized_cumsum_posterior>=1-one_sided_alpha, axis=1, keepdims=True)
    upper_limit_posterior_parameters = np.take_along_axis(sorted_parameters, upper_limit_posterior_indeces, axis=1)
    
    # extract model parameters
    _parameters_upper = upper_limit_posterior_parameters[:,:]

    return _parameters_lower, _parameters_upper

def reduce_to_marginal_credible_intervals(combined_dict, alpha_level, alpha_level_tighter):
    '''
    iterates over all time steps and computes the marginal credible intervals as well as the mean, maximum a posterior and the marginal median of the weight distribution.

    :combined_dict: dict with all the particles and weights that are used to reduce to the median particles
    :alpha_level: ratio of weights outside of the credible interval. e.g. alpha_level=0.05 generates a 95% CI, i.e. 2.5% of weights are above and below the CI.
    alpha_level_tighter: same as alpha_level just for a second (tighter) level.
    '''
    result_dict = {}
    for key in combined_dict.keys():
        data_dict = combined_dict[key]

        result_dict[key] = {}

        median = np.zeros_like(data_dict['para'][:,:,0])
        m5p = np.zeros_like(data_dict['para'][:,:,0])

        maximum_posterior = np.zeros_like(data_dict['para'][:,:,0])

        lower_parameters = np.zeros_like(data_dict['para'][:,:,0])
        upper_parameters = np.zeros_like(data_dict['para'][:,:,0])

        lower_parameters_tight_ci = np.zeros_like(data_dict['para'][:,:,0])
        upper_parameters_tight_ci = np.zeros_like(data_dict['para'][:,:,0])
        
        for t in range(lower_parameters.shape[0]):

            parameter_slice = data_dict['para'][t,:,:]
            weight_slice = data_dict['weights'][t, :, :]

            # marginal medians
            _parameters_lower, _parameters_upper = marginal_credible_intervals(parameter_slice, weights = weight_slice, alpha_level=1)
            _median = (_parameters_lower + _parameters_upper)*0.5
            median[t] = _median.squeeze()

            # max posterior
            max_weight_indeces = np.argmax(data_dict['weights'][t,0,:])
            _max_post = data_dict['para'][t,:,max_weight_indeces]
            maximum_posterior[t] = _max_post

            # marginal 5% and 95%
            _parameters_lower, _parameters_upper = marginal_credible_intervals(parameter_slice, weights = weight_slice, alpha_level=alpha_level)

            lower_parameters[t] = _parameters_lower.squeeze()
            upper_parameters[t] = _parameters_upper.squeeze()

            # marginal 25% and 75%
            _parameters_lower, _parameters_upper = marginal_credible_intervals(parameter_slice, weights = weight_slice, alpha_level=alpha_level_tighter)

            lower_parameters_tight_ci[t] = _parameters_lower.squeeze()
            upper_parameters_tight_ci[t] = _parameters_upper.squeeze()

        mean_post = np.average(data_dict['para'], weights =  data_dict['weights'], axis=-1)
        
        result_dict[key]['median'] = median
        result_dict[key]['mean'] = mean_post
        result_dict[key]['maximum posterior'] = maximum_posterior
        
        result_dict[key]['lower credible interval'] = lower_parameters
        result_dict[key]['upper credible interval'] = upper_parameters

        result_dict[key]['lower credible interval tight ci'] = lower_parameters_tight_ci
        result_dict[key]['upper credible interval tight ci'] = upper_parameters_tight_ci

        print(f'Reduction to CIs done for {key}.')

    return result_dict

def combine_log_evidences(letters, doc_string, doc_count = 1):
    '''
    sums up the log_evdences from the intermediate files.
    Only  works for log_evidences as its addition based.

    :letters: list of letters with which the .h5 files start. Also becomes the keys for the return dict
    :doc_string: ending of the .h5 file which holds the data
    :doc_count: number of documents with the same ending that have to be combined
    '''
    
    combined_dict = {}

    for letter in letters:

        collect_log_evidences = []

        for j in range(doc_count):
            with h5py.File(f"{letter}{doc_string}", "r") as f:

                mc_runs = int((len(list(f.keys()))-1)/2)

                for i in range(mc_runs):

                    mc_string = 'mcgroup_' + str(i)
        
                    '''import the data from the logging files'''
                    log_evidence = f[mc_string]['log_evidence'][:]

                    collect_log_evidences.append(log_evidence)

        combined_log_evidences = np.mean(collect_log_evidences, axis=0)

        combined_dict[letter] = combined_log_evidences
    
    return combined_dict