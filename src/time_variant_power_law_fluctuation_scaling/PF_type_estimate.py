# Copyright (c) YYYY German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Bernhard Jonathan Sattler, Bernhard.sattler@dlr.de
# Date of creation: 2026-10-09

from scipy.stats import norm, gaussian_kde, qmc
import numpy as np
import matplotlib.pyplot as plt


def inverse_continous_uniform(values, _min = 0, _max = 1):
    '''Scales a continuous uniform to a given width.'''
    return_values = _min + values*(_max - _min)

    return return_values

def inverse_2d_independent_gaussian(values, _loc_x = 0, _loc_y = 0, _scale = 1):
    '''Generates samples of two inpedependet Gaussians based on two continous uniform variables.
    This function is based on the Box-Muller-transform.
    :values: np.array of shape (n,2)'''
    standard_normal_x = np.sqrt(-2*np.log(values[:,0]))*np.cos(2*np.pi*values[:,1])
    standard_normal_y = np.sqrt(-2*np.log(values[:,0]))*np.sin(2*np.pi*values[:,1])

    return_x = standard_normal_x*_scale + _loc_x
    return_y = standard_normal_y*_scale + _loc_y

    return_values = np.hstack((return_x[:,None], return_y[:,None]))

    return return_values

def generate_priors(shape = (1000,4), minimals = np.array([0,0]), maximals = np.array([1,1]), gaussian_loc =np.array([0,0]), rng=None):
    '''Generates the Latin hyper cube samples for the priors.'''

    lhc_sampler = qmc.LatinHypercube(d=shape[1], rng = rng)
    sample = lhc_sampler.random(n=shape[0])

    priors = np.zeros_like(sample)

    uniforms = inverse_continous_uniform(values=sample[:,2:], _min=minimals, _max=maximals)
    priors[:,2:] = uniforms

    gaussians = inverse_2d_independent_gaussian(sample[:,:2], _loc_x=gaussian_loc[0], _loc_y=gaussian_loc[1], _scale=np.exp(priors[:,3]))
    priors[:,:2] = gaussians

    return priors

def generate_parameter_space_samples(x_point, y_point,
                                    beta_min, beta_max, log_sigma_min, log_sigma_max,
                                    particle_count, rng):
    '''This function generates a set of possible linear parameters matching a set of
    bi-normally distributed points around a central reference point (x_point,y_point).
    x_point, y_point: reference point
    beta_min, beta_max: range over which beta is uniformly sampled.
    log_sigma_min, log_sigma_max: range over which log_tau is uniformly sampled.
    particle_count: number of generated particles'''

    priors = generate_priors(shape = (particle_count, 4),
                            minimals = np.array([beta_min, log_sigma_min]),
                            maximals = np.array([beta_max, log_sigma_max]),
                            gaussian_loc =np.array([x_point, y_point]),
                            rng=rng)

    # generate the variable indicating the slope (a number of range_samples angles, equally devided between 0° and 90°(pi/2)), excluding 90° as tan(90°) is not defined
    if beta_min != beta_max:
        slope_array = priors[:,2]
    else:
        # estimate a restricted model with beta = 2
        slope_array = beta_min*np.ones_like(priors[:,2])

    # generate the variable indicating the intercept
    intercept_array = priors[:,1] - priors[:,0]*slope_array

    log_observation_noise_scale = priors[:,3]

    # generate parameter space in point form
    parameter_space = np.vstack([intercept_array, slope_array, log_observation_noise_scale])

    return parameter_space

def density_linear_model_analytic(x_point, y_point, intercept, slope, sigma, regression_distance = 'ODR'):
    '''calculates the likelihood of a set of [x, y] points
    under the assumption of dist([x, y], [x, slope*x + intercept]) = norm(mu = 0, sigma = sigma).
    using log(x), log(y) and log(intercept), this estimates a Taylors Law, i.e.
    dist([x, y], [x, intercept*x**slope]) = norm(mu = 0, sigma = sigma).
    Keep in Mind, that Sigma is also in the log-log-space in this case.'''

    if regression_distance == 'ODR_old':
        # The following equation is derived based on the equation
        # slope*x_trace + intercept = (-1/slope) * (x_trace - x) + y
        # which is an analytical derivation of the closest point (x_trace, y_trace) on a 2D linear function
        # to a second point (x, y)
        x_trace = (x_point + slope*y_point - slope*intercept)/(slope**2 + 1)
        y_trace = slope*x_trace + intercept

        # distance based on Pythagoran Theorem
        dist = np.sqrt((x_point-x_trace)**2+(y_point-y_trace)**2)

    elif regression_distance == 'ODR':
        dist = np.abs(slope * x_point - y_point + intercept) / np.sqrt(slope**2 + 1)

    elif regression_distance == 'OLS':
        dist = np.abs(y_point-slope*x_point-intercept)

    # calculate Z-score
    z_score = dist / sigma

    # return likelihood of this distance in a normal distribution analogous to norm.pdf(dist, loc=0, scale=sigma)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z_score**2)

    return pdf

def initialization_data_driven(x_point, y_point, beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count, rng):
    '''Initializes the data space around the prior parameter range and the given observation.'''

    # generate first proposal distribution based on plausible parameter combinations
    # around the first observed value of the time series.
    parameter_space = generate_parameter_space_samples(x_point, y_point, beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count, rng)

    # equal priors for all parameter sets (non-informative/low-information prior)
    parameter_space_length = parameter_space.shape[1]
    model_prior = np.full(parameter_space_length, 1/parameter_space_length)

    return parameter_space, model_prior

def update(parameter_space, model_prior, x_point, y_point):
    '''Updates the particle weghts based on the current observation.'''

    # extract parameters for given parameter set
    _log_alpha = parameter_space[0]
    _beta = parameter_space[1]
    _log_sigma = parameter_space[2]
    _sigma = np.exp(_log_sigma)

    # likelihoods and Z-scores of all models
    model_likelihoods = density_linear_model_analytic(x_point, y_point, _log_alpha, _beta, _sigma)

    evidence = np.dot(model_likelihoods, model_prior)

    temp = model_likelihoods * model_prior

    posterior = temp/evidence

    return evidence, posterior

def add_prior_particles(parameters, x_point, y_point, beta_min, beta_max, log_sigma_min, log_sigma_max, proportion_of_prior, rng):
    '''Subsitutes a part of the current state estimate by particles generated from the data based prior assumption.'''

    particle_count = len(parameters[0])
    prior_particles, prior_weights = initialization_data_driven(x_point, y_point, beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count, rng)

    num_inserted_prior_particles = int(proportion_of_prior*particle_count)
    random_selection_mask = np.zeros(particle_count, dtype=bool)
    random_selection_mask[rng.choice(particle_count, size=num_inserted_prior_particles, replace=False)] = True
    parameters[:,random_selection_mask] = prior_particles[:,random_selection_mask]

    return parameters

def resample(x_point, y_point, beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count,
             parameter_space, posterior, ess_threshold, proportion_of_prior, rng):
    
    '''Resampling is aimed to prevent sample impoverishment and degeneracy'''

    # effective sample size to judge, if the particles are sufficiently used, i.e. if there are not too many unnecessary particles
    effective_sample_size = 1/np.sum(np.square(posterior))

    # resample parameter space if the effective sample size is too small,
    # i.e. when effictively only half of the points (parameter sets) contribute to the estimate of the parameter
    # -> imminent particle impoverishment
    if effective_sample_size<ess_threshold*len(posterior):


        if beta_min != beta_max:

            kde = gaussian_kde(parameter_space, weights = posterior)
            parameter_space = kde.resample(particle_count, seed=rng)
            model_prior = np.ones_like(posterior)* (1/particle_count)

            parameter_space = add_prior_particles(parameter_space, x_point, y_point, beta_min, beta_max, log_sigma_min, log_sigma_max, proportion_of_prior, rng)
            
        else:
            # select only log_alpha and tau parameters to allow gaussian kde
            parameter_space_without_beta = parameter_space[[0, 2], :]

            kde = gaussian_kde(parameter_space_without_beta, weights = posterior)
            parameter_space_without_beta = kde.resample(particle_count, seed=rng)

            # re-insert beta from original parameter space
            parameter_space = np.insert(parameter_space_without_beta, 1, parameter_space[1], axis=0)

            model_prior = np.ones_like(posterior)* (1/particle_count)

            parameter_space = add_prior_particles(parameter_space, x_point, y_point, beta_min, beta_max, log_sigma_min, log_sigma_max, proportion_of_prior, rng)

    else:
        model_prior=posterior

    return parameter_space, model_prior

def estimate_taylors_law(mean_, var_, beta_min, beta_max, log_sigma_min, log_sigma_max, ess_threshold, proportion_of_prior,
                        particle_count, rng):

    '''Main filter code. Runs the steps of the filter according to the description in the paper.'''

    length = len(mean_)

    # create log-log-scaling of the data by taking the logarithm of both value streams
    log_mean_ = np.log(mean_)
    log_var_ = np.log(var_)

    # initialization
    parameter_space, model_prior = initialization_data_driven(log_mean_[0], log_var_[0], beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count, rng)
    
    # set up logging (in memory)
    log_evidence_return = np.zeros(length)
    posterior_map_return = np.zeros((length, parameter_space.shape[0], parameter_space.shape[1])) # The second dimension is redudant, as the posterior is identical for log_alpha and beta, but this enables easier data handling
    parameter_space_map_return = np.zeros((length, parameter_space.shape[0], parameter_space.shape[1]))

    # the following loop iterates through the time series. by adjusting the range, we can test the time series partially
    for time_step_i in range(length):
        
        '''Please mind, that the prediction step is skipped in the implementation, as the system state is assumed to be stationary (except for the process noise),
         and the process noise is modeled implicitly in the resampling step.'''
        
        # log parameter space into the buffer
        parameter_space_map_return[time_step_i] = parameter_space

        log_mean_t = log_mean_[time_step_i]
        log_var_t = log_var_[time_step_i]

        # updating the estimate on the basis of the new observation
        evidence, posterior = update(parameter_space, model_prior, log_mean_t, log_var_t)

        '''This section logs the obtained filter step results'''
        log_evidence_return[time_step_i] = np.log(evidence)
        posterior_map_return[time_step_i] = posterior

        ''' If particle diversity, measured by effective sample size, is too small, we resample the parameter space.        
        After this step, we have a new prior and parameter space for the next step'''

        parameter_space, model_prior = resample(log_mean_t, log_var_t, beta_min, beta_max, log_sigma_min, log_sigma_max, particle_count,
                                            parameter_space, posterior, ess_threshold, proportion_of_prior, rng)

    return [log_evidence_return, posterior_map_return, parameter_space_map_return]

