# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# Trillium Technologies
# University of Oxford
# Giacomo Acciarini (giacomo.acciarini@gmail.com)
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.

import dsgp4
import numpy as np
import uuid
import torch
import pyro
import pyro.distributions as dist

from . import GNSS, Radar, ConjunctionDataMessage, util
from dsgp4 import TLE

from pyro.distributions import MixtureSameFamily, Categorical, Uniform, Normal, Bernoulli

def find_conjunction(tr0, 
                     tr1, 
                     miss_dist_threshold):
    """"
    Find the closest and first conjunction between two trajectories.
    Args:
        tr0 (``torch.Tensor``): The first trajectory.
        tr1 (``torch.Tensor``): The second trajectory.
        miss_dist_threshold (``float``): The maximum distance between the trajectories to be considered a conjunction.
    
    Returns:
        i_min (``int``): The index of the closest conjunction.
        d_min (``float``): The distance of the closest conjunction.
        i_conj (``int``): The index of the first conjunction below `miss_dist_threshold`.
        d_conj (``float``): The distance of the closest conjunction in tr1.
    """
    d = tr0 - tr1
    squared_norm = torch.einsum('ix,ix->i', d, d)

    i_min = int(torch.argmin(squared_norm))
    d_min = squared_norm[i_min].sqrt()

    below_threshold = squared_norm < miss_dist_threshold**2
    below_threshold_indices = below_threshold.nonzero().flatten()
    if below_threshold_indices.nelement() == 0:
        return i_min, d_min, None, None
    else:
        i_conj = int(below_threshold_indices[0])
        d_conj = squared_norm[i_conj].sqrt()
        return i_min, d_min, i_conj, d_conj

def default_prior():
    """
    This function returns a dictionary of TLE elements priors.
    Each prior is a probability density function defined using Pyro distributions.
    The population of objects from which these priors were derived is the one of May 2022.
    The priors are defined as follows:
        - mean_motion_prior: MixtureSameFamily
        - mean_anomaly_prior: Uniform
        - eccentricity_prior: MixtureSameFamily
        - inclination_prior: MixtureSameFamily
        - argument_of_perigee_prior: Uniform
        - raan_prior: Uniform
        - mean_motion_first_derivative_prior: Uniform
        - b_star_prior: MixtureSameFamily
    The parameters of the distributions are based on the population of objects.
    The priors are defined in the following ranges:
        - mean_motion: [0.0, 0.004]
        - mean_anomaly: [0.0, 2 * pi]
        - eccentricity: [0.0, 0.9]
        - inclination: [0.0, pi]
        - argument_of_perigee: [0.0, 2 * pi]
        - raan: [0.0, 2 * pi]
        - mean_motion_first_derivative: [4.937096738377722e-13, 5.807570136601159e-13]
        - b_star: (-inf, inf)
    """
    p = {}

    # Mean Motion Prior
    p['mean_motion_prior'] = MixtureSameFamily(
        mixture_distribution=Categorical(probs=torch.tensor([
            0.12375596165657043, 0.05202080309391022, 0.21220888197422028,
            0.0373813770711422, 0.01674230769276619, 0.5578906536102295])),
        component_distribution=util.TruncatedNormal(
            low=0.0, high=0.004, loc=torch.tensor([
                0.0010028142482042313, 0.00017592836171388626, 0.0010926761478185654,
                0.0003353552892804146, 0.0007777251303195953, 0.001032940074801445]),
            scale=torch.tensor([
                0.00004670943133533001, 0.00003172305878251791, 0.000027726569678634405,
                0.00007733114063739777, 0.00013636205345392227, 0.00002651428570970893]))
    )

    # Mean Anomaly Prior
    p['mean_anomaly_prior'] = Uniform(low=0.0, high=2 * torch.pi)

    # Eccentricity Prior
    p['eccentricity_prior'] = MixtureSameFamily(
        mixture_distribution=Categorical(probs=torch.tensor([
            0.5433819890022278, 0.04530993849039078, 0.08378008753061295,
            0.02705608867108822, 0.03350389748811722, 0.2669680118560791])),
        component_distribution=util.TruncatedNormal(
            low=0.0, high=0.8999999761581421, loc=torch.tensor([
                0.0028987403493374586, 0.6150050163269043, 0.05085373669862747,
                0.3420163094997406, 0.7167646288871765, 0.013545362278819084]),
            scale=torch.tensor([
                0.002526970813050866, 0.07872536778450012, 0.024748045951128006,
                0.18968918919563293, 0.011966796591877937, 0.0068586356937885284]))
    )

    # Inclination Prior
    p['inclination_prior'] = MixtureSameFamily(
        mixture_distribution=Categorical(probs=torch.tensor([
            0.028989605605602264, 0.10272273421287537, 0.02265254408121109, 0.019256576895713806,
            0.028676774352788925, 0.06484941393136978, 0.13786117732524872, 0.0010146398562937975,
            0.047179922461509705, 0.01607278548181057, 0.020023610442876816, 0.06644929945468903,
            0.4442509114742279])),
        component_distribution=util.TruncatedNormal(
            low=0.0, high=torch.pi, loc=torch.tensor([
                0.09954200685024261, 1.4393062591552734, 1.736578106880188, 1.0963480472564697,
                0.48166394233703613, 0.9063634872436523, 1.275956392288208, 2.5208728313446045,
                1.5189905166625977, 0.3474450707435608, 0.6648743152618408, 1.1465401649475098,
                1.7207987308502197]),
            scale=torch.tensor([
                0.04205162078142166, 0.012214339338243008, 0.11822951585054398, 0.010178830474615097,
                0.04073172062635422, 0.04156989976763725, 0.02754846028983593, 0.003279004478827119,
                0.02461068518459797, 0.0433642603456974, 0.11472384631633759, 0.014345825649797916,
                0.012212350033223629]))
    )

    # Argument of Perigee Prior
    p['argument_of_perigee_prior'] = Uniform(low=0.0, high=2 * torch.pi)

    # RAAN Prior
    p['raan_prior'] = Uniform(low=0.0, high=2 * torch.pi)

    # Mean Motion First Derivative Prior
    p['mean_motion_first_derivative_prior'] = Uniform(4.937096738377722e-13, 5.807570136601159e-13)

    # B* Prior
    p['b_star_prior'] = MixtureSameFamily(
        mixture_distribution=Categorical(probs=torch.tensor([
            0.9688150882720947, 0.0012630978599190712, 0.024090370163321495, 0.0009446446783840656,
            0.0048867943696677685])),
        component_distribution=Normal(
            loc=torch.tensor([0.0002002030232688412, 0.3039868175983429, 0.003936616238206625,
                              -0.04726288095116615, 0.08823495358228683]),
            scale=torch.tensor([0.0011279708705842495, 0.06403032690286636, 0.012939595617353916,
                                0.17036935687065125, 0.061987943947315216]))
    )

    return p


class Conjunction:
    """
    This class implements the Conjunction class, which is used to generate conjunction data messages (CDM) for two objects in space.
    The class uses the Pyro probabilistic programming library to define the model and perform inference.
    The class has the following attributes:
        - time0: The time of the first observation in MJD.
        - max_duration_days: The maximum duration of the simulation in days.
        - time_resolution: The time resolution of the simulation in seconds.
        - time_upsample_factor: The upsample factor for the time resolution.
        - miss_dist_threshold: The miss distance threshold in meters.
        - prior_dict: A dictionary containing the priors for the TLE elements.
        - t_prob_new_obs: The probability of a new observation for the target.
        - c_prob_new_obs: The probability of a new observation for the chaser.
        - cdm_update_every_hours: The interval at which to update the CDM in hours.
        - mc_samples: The number of Monte Carlo samples to use for uncertainty propagation.
        - mc_upsample_factor: The upsample factor for the Monte Carlo samples.
        - pc_method: The method to use for calculating the probability of collision.
        - collision_threshold: The threshold for considering a collision in meters.
        - likelihood_t_stddev: The standard deviation of the likelihood for the target.
        - likelihood_c_stddev: The standard deviation of the likelihood for the chaser.
        - likelihood_time_to_tca_stddev: The standard deviation of the likelihood for time to TCA.

    Example:
        >>> from kessler import Conjunction
        >>> conj = Conjunction()
    """
    def __init__(self,
                 time0=58991.90384230018,
                 max_duration_days=7.0,
                 time_resolution=6e5,
                 time_upsample_factor=100,
                 miss_dist_threshold=5e3,
                 prior_dict=None,
                 t_prob_new_obs=0.96,
                 c_prob_new_obs=0.4,
                 cdm_update_every_hours=8.,
                 mc_samples=100,
                 mc_upsample_factor=100,
                 pc_method='MC',
                 up_method='MC',
                 collision_threshold=70,
                 likelihood_t_stddev=[3.71068006e+02, 9.99999999e-02, 1.72560879e-01],
                 likelihood_c_stddev=[3.71068006e+02, 9.99999999e-02, 1.72560879e-01],
                 likelihood_time_to_tca_stddev=0.7,
                 t_observing_instruments=None,
                 c_observing_instruments=None,
                 t_tle=None,
                 c_tle=None):
        self._time0 = time0
        self._max_duration_days = max_duration_days
        self._time_resolution = time_resolution
        self._time_upsample_factor = time_upsample_factor
        self._delta_time = max_duration_days / time_resolution
        self._miss_dist_threshold = miss_dist_threshold  # miss distance threshold in [m]
        self._prior_dict = prior_dict or default_prior()
        self._t_prob_new_obs = t_prob_new_obs
        self._c_prob_new_obs = c_prob_new_obs
        self._cdm_update_every_hours = cdm_update_every_hours
        self._mc_samples = mc_samples
        self._mc_upsample_factor = mc_upsample_factor
        if pc_method not in ['MC', 'FOSTER-1992']:
            raise ValueError(
                f"Unknown method for probability of collision: {pc_method}. Currently, we only support MC and FOSTER-1992.\n"
                "We are happy to receive your contributions through pull requests to extend Kessler's support to other Pc methods.")        
        self._pc_method = pc_method
        if up_method not in ['MC', 'STM']:
            raise ValueError(
                f"Unknown method for uncertainty propagation: {up_method}. Currently, we only support MC and STM.\n"
                "We are happy to receive your contributions through pull requests to extend Kessler's support to other UP methods.")
        self._up_method = up_method
        self._collision_threshold = collision_threshold
        self._likelihood_t_stddev = likelihood_t_stddev
        self._likelihood_c_stddev = likelihood_c_stddev
        self._likelihood_time_to_tca_stddev = likelihood_time_to_tca_stddev
        if t_observing_instruments is None:
            t_instrument_characteristics={'bias_xyz': np.array([[0., 0., 0.],[0., 0., 0.]]), 'covariance_rtn': np.array([1e-9, 1.115849341564346, 0.059309835843067, 1e-9, 1e-9, 1e-9])**2}
            t_observing_instruments=[GNSS(t_instrument_characteristics)]
            print(f'No observing instruments for target, using default one with diagonal covariance {t_observing_instruments[0]._instrument_characteristics['covariance_rtn']}')
        if c_observing_instruments is None:
            c_instrument_characteristics={'bias_xyz': np.array([[0., 0., 0.],[0., 0., 0.]]), 'covariance_rtn': np.array([1.9628939405514678, 2.2307686944695706, 0.9660907831563862, 1e-9, 1e-9, 1e-9])**2}
            c_observing_instruments=[Radar(c_instrument_characteristics)]
            print(f'No observing instruments for chaser, using default one with diagonal covariance {c_observing_instruments[0]._instrument_characteristics['covariance_rtn']}')
        if len(t_observing_instruments) == 0 or len(c_observing_instruments) == 0:
            raise ValueError("We need at least one observing instrument for target and chaser!")
        self._t_observing_instruments = t_observing_instruments
        self._c_observing_instruments = c_observing_instruments
        self._t_tle = t_tle
        self._c_tle = c_tle

    def make_target(self):
        if self._t_tle is None:
            d = {}
            d['mean_motion'] = pyro.sample('t_mean_motion', self._prior_dict['mean_motion_prior'])
            d['mean_anomaly'] = pyro.sample('t_mean_anomaly', self._prior_dict['mean_anomaly_prior'])
            d['eccentricity'] = pyro.sample('t_eccentricity', self._prior_dict['eccentricity_prior'])
            d['inclination'] = pyro.sample('t_inclination', self._prior_dict['inclination_prior'])
            d['argument_of_perigee'] = pyro.sample('t_argument_of_perigee', self._prior_dict['argument_of_perigee_prior'])
            d['raan'] = pyro.sample('t_raan', self._prior_dict['raan_prior'])
            d['mean_motion_first_derivative'] = pyro.sample('t_mean_motion_first_derivative', self._prior_dict['mean_motion_first_derivative_prior'])
            d['mean_motion_second_derivative'] = 0.0
            pyro.deterministic('t_mean_motion_second_derivative', torch.tensor(d['mean_motion_second_derivative']))
            d['b_star'] = pyro.sample('t_b_star', self._prior_dict['b_star_prior'])
            d['satellite_catalog_number'] = 43437
            d['classification'] = 'U'
            d['international_designator'] = '18100A'
            d['ephemeris_type'] = 0
            d['element_number'] = 9996
            d['revolution_number_at_epoch'] = 56353
            d['epoch_year'] = dsgp4.util.from_mjd_to_datetime(self._time0).year
            d['epoch_days'] = dsgp4.util.from_mjd_to_epoch_days_after_1_jan(self._time0)
            tle = TLE(d)
            return tle
        else:
            mean_anomaly = pyro.sample('t_mean_anomaly', self._prior_dict['mean_anomaly_prior'])
            tle = self._t_tle.copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyro.deterministic('t_mean_motion', torch.tensor(tle.mean_motion))
            pyro.deterministic('t_eccentricity', torch.tensor(tle.eccentricity))
            pyro.deterministic('t_inclination', torch.tensor(tle.inclination))
            pyro.deterministic('t_argument_of_perigee', torch.tensor(tle.argument_of_perigee))
            pyro.deterministic('t_raan', torch.tensor(tle.raan))
            pyro.deterministic('t_mean_motion_first_derivative', torch.tensor(tle.mean_motion_first_derivative))
            pyro.deterministic('t_mean_motion_second_derivative', torch.tensor(tle.mean_motion_second_derivative))
            pyro.deterministic('t_b_star', torch.tensor(tle.b_star))
            return tle
    def make_chaser(self):
        """
        This function creates a chaser object, as a TLE (``dsgp4.tle.TLE``).
        """
        if self._c_tle is None:
            d = {}
            d['mean_motion'] = pyro.sample('c_mean_motion',self._prior_dict['mean_motion_prior'])
            d['mean_anomaly'] = pyro.sample('c_mean_anomaly',self._prior_dict['mean_anomaly_prior'])
            d['eccentricity'] = pyro.sample('c_eccentricity',self._prior_dict['eccentricity_prior'])
            d['inclination'] = pyro.sample('c_inclination',self._prior_dict['inclination_prior'])
            d['argument_of_perigee'] = pyro.sample('c_argument_of_perigee',self._prior_dict['argument_of_perigee_prior'])
            d['raan'] = pyro.sample('c_raan',self._prior_dict['raan_prior'])
            d['mean_motion_first_derivative'] = pyro.sample('c_mean_motion_first_derivative',self._prior_dict['mean_motion_first_derivative_prior'])
            d['mean_motion_second_derivative'] = 0.0  # pybrob.sample(Uniform(0.0,1e-17))
            pyro.deterministic('c_mean_motion_second_derivative',torch.tensor(d['mean_motion_second_derivative']))
            d['b_star'] = pyro.sample('c_b_star',self._prior_dict['b_star_prior'])
            d['satellite_catalog_number'] = 43437
            d['classification'] = 'U'
            d['international_designator'] = '18100A'
            d['ephemeris_type'] = 0
            d['element_number'] = 9996
            d['revolution_number_at_epoch'] = 56353
            d['epoch_year'] = dsgp4.util.from_mjd_to_datetime(self._time0).year
            d['epoch_days'] = dsgp4.util.from_mjd_to_epoch_days_after_1_jan(self._time0)
            tle = TLE(d)
            return tle
        else:
            mean_anomaly = pyro.sample('c_mean_anomaly', self._prior_dict['mean_anomaly_prior'])
            tle = self._c_tle.copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyro.deterministic('c_mean_motion',tle.mean_motion)
            pyro.deterministic('c_eccentricity',tle.eccentricity)
            pyro.deterministic('c_inclination',tle.inclination)
            pyro.deterministic('c_argument_of_perigee',tle.argument_of_perigee)
            pyro.deterministic('c_raan',tle.raan)
            pyro.deterministic('c_mean_motion_first_derivative',tle.mean_motion_first_derivative)
            pyro.deterministic('c_mean_motion_second_derivative',tle.mean_motion_second_derivative)
            pyro.deterministic('c_b_star',tle.b_star)
            return tle

    def generate_cdm(self, 
                     t_state_new_obs, 
                     c_state_new_obs, 
                     time_obs_mjd, 
                     time_conj_mjd, 
                     t_tle, 
                     c_tle, 
                     previous_cdm):
        """
        This function generates a conjunction data message (``kessler.cdm.ConjunctionDataMessage``) from the current state of the chaser and target.
        
        Args:
            t_state_new_obs (``torch.tensor``): The state of the target at the time of the observation.
            c_state_new_obs (``torch.tensor``): The state of the chaser at the time of the observation.
            time_obs_mjd (``float``): The time of the CDM in MJD.
            time_conj_mjd (``float``): The time of the conjunction in MJD.
            t_tle (``dsgp4.tle.TLE``): The TLE of the target.
            c_tle (``dsgp4.tle.TLE``): The TLE of the chaser.
            previous_cdm (``kessler.cdm.ConjunctionDataMessage``): The previous conjunction data message (it can be None in case there's no history of CDMs).
        
        Returns:
            cdm (``kessler.cdm.ConjunctionDataMessage``): The conjunction data message. None, if no CDM has to be generated.
        """
        if c_state_new_obs is not None or t_state_new_obs is not None:
            if previous_cdm:
                cdm = previous_cdm.copy()
            else:
                cdm = ConjunctionDataMessage()
                cdm.set_object(0, 'OBJECT_DESIGNATOR', 'KESSLER_SOFTWARE_'+str(uuid.uuid1()))
                cdm.set_object(1, 'OBJECT_DESIGNATOR', 'KESSLER_SOFTWARE_'+str(uuid.uuid1()))

                cdm.set_header('ORIGINATOR', 'KESSLER_SOFTWARE')

                if self._t_tle is not None or hasattr(self, '_tles'):
                    cdm.set_object(0, 'OBJECT_DESIGNATOR', t_tle.international_designator)
                    cdm.set_object(0, 'INTERNATIONAL_DESIGNATOR', t_tle.international_designator)
                    if hasattr(t_tle,"name"):
                        cdm.set_object(0, 'OBJECT_NAME', t_tle.name)
                    else:
                        cdm.set_object(0, 'OBJECT_NAME', 'KESSLER_SOFTWARE_TARGET')
                    cdm.set_object(0, 'CATALOG_NAME', t_tle.satellite_catalog_number)
                else:
                    cdm.set_object(0, 'OBJECT_DESIGNATOR', 'KESSLER_SOFTWARE_'+str(uuid.uuid1()))
                    cdm.set_object(0, 'INTERNATIONAL_DESIGNATOR', 'UNKNOWN')
                    cdm.set_object(0, 'OBJECT_NAME', 'KESSLER_SOFTWARE_TARGET')
                    cdm.set_object(0, 'CATALOG_NAME', 'UNKNOWN')
                cdm.set_object(0, 'EPHEMERIS_NAME', 'NONE')
                cdm.set_object(0, 'COVARIANCE_METHOD', 'CALCULATED')
                cdm.set_object(0, 'MANEUVERABLE', 'N/A')
                cdm.set_object(0, 'REF_FRAME', 'ITRF')
                cdm.set_object(0, 'ORBIT_CENTER', 'EARTH')

                if self._c_tle is not None or hasattr(self, '_tles'):
                    cdm.set_object(1, 'OBJECT_DESIGNATOR', c_tle.international_designator)
                    cdm.set_object(1, 'INTERNATIONAL_DESIGNATOR', c_tle.international_designator)
                    if hasattr(c_tle,"name"):
                        cdm.set_object(1, 'OBJECT_NAME', c_tle.name)
                    else:
                        cdm.set_object(1, 'OBJECT_NAME', 'KESSLER_SOFTWARE_CHASER')
                    cdm.set_object(1, 'CATALOG_NAME', c_tle.satellite_catalog_number)
                else:
                    cdm.set_object(1, 'OBJECT_DESIGNATOR', 'KESSLER_SOFTWARE_'+str(uuid.uuid1()))
                    cdm.set_object(1, 'INTERNATIONAL_DESIGNATOR', 'UNKNOWN')
                    cdm.set_object(1, 'OBJECT_NAME', 'KESSLER_SOFTWARE_CHASER')
                    cdm.set_object(1, 'CATALOG_NAME', 'UNKNOWN')
                cdm.set_object(1, 'EPHEMERIS_NAME', 'NONE')
                cdm.set_object(1, 'COVARIANCE_METHOD', 'CALCULATED')
                cdm.set_object(1, 'MANEUVERABLE', 'N/A')
                cdm.set_object(1, 'REF_FRAME', 'ITRF')
                cdm.set_object(1, 'ORBIT_CENTER', 'EARTH')

            cdm.set_header('MESSAGE_ID', 'KESSLER_SOFTWARE_{}'.format(str(uuid.uuid1())))

            tca_jd = util.from_mjd_to_jd(time_conj_mjd)
            obs_jd = util.from_mjd_to_jd(time_obs_mjd)
            cdm.set_relative_metadata('TCA', util.from_jd_to_cdm_datetime_str(tca_jd))
            cdm.set_header('CREATION_DATE', util.from_jd_to_cdm_datetime_str(obs_jd))
            if t_state_new_obs is not None:
                obs_tle_t,_=dsgp4.newton_method(t_tle,time_obs_mjd,verbose=False)
                # we return the state in XYZ, the state in RTN and the cov matrix in RTN
                if self._up_method == 'MC':
                    t_mean_state_tca_xyz_TEME, t_cov_state_tca_rtn = self.propagate_uncertainty_monte_carlo(state_xyz = t_state_new_obs,
                                                                                                            cov_matrix_diagonal_rtn = self._t_cov_matrix_diagonal_obs_noise,
                                                                                                            time_ca_mjd = time_conj_mjd,
                                                                                                            obs_tle = obs_tle_t)
                    t_mean_state_tca_xyz_ITRF = util.from_TEME_to_ITRF(t_mean_state_tca_xyz_TEME, tca_jd)/ 1e3
                    cdm.set_state(0, t_mean_state_tca_xyz_ITRF)
                    cdm.set_covariance(0, t_cov_state_tca_rtn)
                #elif self._up_method == 'STM':
            if c_state_new_obs is not None:
                obs_tle_c,_=dsgp4.newton_method(c_tle,time_obs_mjd,verbose=False)
                if self._up_method == 'MC':
                    c_mean_state_tca_xyz_TEME, c_cov_state_tca_rtn = self.propagate_uncertainty_monte_carlo(state_xyz = c_state_new_obs,
                                                                                                            cov_matrix_diagonal_rtn = self._c_cov_matrix_diagonal_obs_noise,
                                                                                                            time_ca_mjd = time_conj_mjd,
                                                                                                            obs_tle = obs_tle_c)
                    c_mean_state_tca_xyz_ITRF = util.from_TEME_to_ITRF(c_mean_state_tca_xyz_TEME, tca_jd)/ 1e3
                    cdm.set_state(1, c_mean_state_tca_xyz_ITRF)
                    cdm.set_covariance(1, c_cov_state_tca_rtn)
                #elif self._up_method == 'STM':
                    #in this case we propagate the covariances using the state transition matrix (first-order approximation)
            #Recommended standards to compute Pc:
            #FOSTER-1992, CHAN-1997,PATERA-2001, and, ALFANO-2005
            if self._pc_method == 'MC':
                #We extract the state and transform it to RTN
                t_state_tca_rtn, _ = util.from_cartesian_to_rtn(cdm.get_state(0)*1e3)
                c_state_tca_rtn, _ = util.from_cartesian_to_rtn(cdm.get_state(1)*1e3)
                #and we extract the covariance matrix in the RTN frame:
                t_cov_pos_tca=cdm.get_covariance(0)[:3,:3]
                c_cov_pos_tca=cdm.get_covariance(1)[:3,:3]
                #we now sample the state of the target and chaser at the time of closest approach
                t_samples_tca=torch.distributions.MultivariateNormal(torch.tensor(t_state_tca_rtn[0]),torch.tensor(t_cov_pos_tca)).sample((int(self._mc_samples*self._mc_upsample_factor),))
                c_samples_tca=torch.distributions.MultivariateNormal(torch.tensor(c_state_tca_rtn[0]),torch.tensor(c_cov_pos_tca)).sample((int(self._mc_samples*self._mc_upsample_factor),))
                #we compute all vs all miss distances:
                miss_distances=torch.cdist(t_samples_tca,c_samples_tca)
                #we compute the probability of collision as the number of samples that are below the threshold
                probability_of_collision=(miss_distances<self._collision_threshold).to(torch.int32).sum()/torch.numel(miss_distances)
            #elif self._pc_method == 'FOSTER-1992':
            cdm.set_relative_metadata('COLLISION_PROBABILITY', float(probability_of_collision))
            cdm.set_relative_metadata('COLLISION_PROBABILITY_METHOD', self._pc_method)
            return cdm
        else:
            return None
        
    def propagate_uncertainty_monte_carlo(self, state_xyz, cov_matrix_diagonal_rtn, time_ca_mjd, obs_tle):
        """
        This function propagates the uncertainty of the state of the target at the time of observation, using Monte Carlo sampling.

        Args:
            state_xyz (``np.array``): state of the target at the time of observation in cartesian coordinates
            cov_matrix_diagonal_rtn (``np.array``): diagonal of the covariance matrix of the state of the target at the time of observation in RTN coordinates
            time_ca_mjd (``float``): time of closest approach in MJD
            obs_tle (``dsgp4.TLE``): TLE of the target at the time of observation

        Returns:
            mean_state_tca_xyz_TEME (``np.array``): mean state of the target at the time of closest approach in cartesian coordinates in TEME frame
            cov_state_tca_rtn (``np.array``): covariance matrix of the state of the target at the time of closest approach in RTN coordinates
        """
        
        #I transform the covariance from RTN to XYZ
        state_rtn, cartesian_to_rtn_rotation_matrix = util.from_cartesian_to_rtn(state_xyz)

        #I construct the 6x6 rotation matrix from cartesian -> RTN
        transformation_matrix_cartesian_to_rtn = np.zeros((6,6))
        transformation_matrix_cartesian_to_rtn[0:3, 0:3] = cartesian_to_rtn_rotation_matrix
        transformation_matrix_cartesian_to_rtn[3:,3:] = cartesian_to_rtn_rotation_matrix
        C_xyz = np.matmul(np.matmul(transformation_matrix_cartesian_to_rtn.T, np.diag(cov_matrix_diagonal_rtn)),transformation_matrix_cartesian_to_rtn)

        ###### Similarity transformation: Cartesian Covariance -> TLE Covariance ######
        state_=torch.tensor(state_xyz)
        #we extract the partials of TLE elements w.r.t. cartesian coordinates
        dtle_dx=util.keplerian_cartesian_partials(state_.requires_grad_(), self._mu_earth)
        #we construct the covariance matrix of the TLE elements via similarity transformation, using the partials of TLE elements w.r.t. cartesian coordinates
        Cov_tle=np.matmul(np.matmul(dtle_dx,C_xyz),dtle_dx.T)
        if ~torch.all(torch.from_numpy(np.real(np.linalg.eig(Cov_tle)[0])) > 1e-11):
            #Covariance matrix is not sdp, forcing symmetry and adding a small value to the diagonal
            Cov_tle = 0.5 * (Cov_tle + Cov_tle.T)
            Cov_tle += 1e-10 * np.eye(Cov_tle.shape[0])
        #we extract the mean TLE elements from the TLE object, that will be used for the sampling
        mean_tle_els=torch.tensor([obs_tle.semi_major_axis,obs_tle._ecco,obs_tle._inclo,obs_tle._nodeo,obs_tle._argpo,obs_tle._mo])

        ###### TLEs construction and propagation from the given samples ######
        tle_data={}
        tle_data['mean_motion_first_derivative'] = obs_tle.mean_motion_first_derivative
        tle_data['mean_motion_second_derivative'] = obs_tle.mean_motion_second_derivative
        tle_data['epoch_year'] = obs_tle.epoch_year
        tle_data['epoch_days'] = obs_tle.epoch_days
        tle_data['b_star'] = obs_tle.b_star
        tle_data['satellite_catalog_number'] = obs_tle.satellite_catalog_number
        tle_data['classification'] = obs_tle.classification
        tle_data['international_designator'] = obs_tle.international_designator
        tle_data['ephemeris_type'] = obs_tle.ephemeris_type
        tle_data['element_number'] = obs_tle.element_number
        tle_data['revolution_number_at_epoch'] = obs_tle.revolution_number_at_epoch

        try:
            dist=torch.distributions.MultivariateNormal(loc=mean_tle_els,covariance_matrix=torch.tensor(Cov_tle))
            samples=dist.sample((self._mc_samples,))
            #we convert all the sampled semi-major axis into mean motion:
            samples[:,0]= (self._mu_earth/samples[:,0]**3)**(0.5)
        except Exception as e:
            if "Expected parameter covariance_matrix" in str(e):
                if mean_tle_els[1]<0.:
                    tle_data['eccentricity']=torch.tensor(0.)#mean_tle_els[0]
                else:
                    tle_data['eccentricity']=mean_tle_els[1]
                tle_data['argument_of_perigee']=mean_tle_els[4]
                tle_data['inclination']=mean_tle_els[2]
                tle_data['mean_anomaly']=mean_tle_els[5]
                tle_data['mean_motion']=(self._mu_earth/mean_tle_els[0]**3)**(1./2.)
                tle_data['raan']=mean_tle_els[3]
                # Propagate object at tca
                tle_object = TLE(tle_data)
                dsgp4.initialize_tle(tle_object)
                tsinces=(torch.tensor(time_ca_mjd)-dsgp4.util.from_datetime_to_mjd(tle_object._epoch))*1440.
                mean_state_tca_xyz_TEME=dsgp4.propagate(tle_object, tsinces).detach().numpy()*1e3
                return mean_state_tca_xyz_TEME, np.diag(cov_matrix_diagonal_rtn)
            else:
                raise e

        mc_states_tca_xyz=[]
        tle_objects=[]
        for sample in samples:
            if sample[1]<0.:
                tle_data['eccentricity']=torch.tensor(0.)
            else:
                tle_data['eccentricity']=sample[1]
            tle_data['argument_of_perigee']=sample[4]
            tle_data['inclination']=sample[2]
            tle_data['mean_anomaly']=sample[5]
            tle_data['mean_motion']=sample[0]
            tle_data['raan']=sample[3]
            # Propagate object at tca
            tle_objects.append(TLE(tle_data).copy())
        try:
            _,tle_batch=dsgp4.initialize_tle(tle_objects)
        except Exception as e:
            pyro.deterministic('conj',torch.tensor(False))
            return
        tsinces=torch.stack([(torch.tensor(time_ca_mjd)-dsgp4.util.from_datetime_to_mjd(tle_objects[0]._epoch))*1440.]*len(tle_objects))
        mc_states_tca_xyz=dsgp4.propagate_batch(tle_batch,tsinces).numpy()*1e3
        #torch.random.set_rng_state(rng_state)
        #mc_states_tca_xyz = np.stack(mc_states_tca_xyz)
        mean_state_tca_xyz_TEME = mc_states_tca_xyz.mean(axis=0)
        rotation_matrix_tca = util.rotation_matrix(mean_state_tca_xyz_TEME)

        mc_states_tca_rtn = np.zeros_like(mc_states_tca_xyz)
        for i, mc_state_tca_xyz in enumerate(mc_states_tca_xyz):
            mc_states_tca_rtn[i], _ = util.from_cartesian_to_rtn(mc_state_tca_xyz, rotation_matrix_tca)
        cov_state_tca_rtn = np.cov(mc_states_tca_rtn.reshape(-1, 6).transpose())
        return mean_state_tca_xyz_TEME, cov_state_tca_rtn
    

    def forward(self):
        # Create the target & chaser:
        t_tle = self.make_target()
        c_tle = self.make_chaser()
        #we immediately exclude cases where target and chaser are the same object:
        # if t_tle.international_designator==c_tle.international_designator:
        #     pyro.deterministic('conj',torch.tensor(False))
        #     return
        # pyro.deterministic('t_tle0_line1',t_tle.line1)
        # pyro.deterministic('t_tle0_line2',t_tle.line2)
        # pyro.deterministic('c_tle0_line1',c_tle.line1)
        # pyro.deterministic('c_tle0_line2',c_tle.line2)
        #we use a perigee/apogee filter to immediately exclude cases
        if (((t_tle.apogee_alt()+self._miss_dist_threshold+1000)<c_tle.perigee_alt())==True) or (((c_tle.apogee_alt()+self._miss_dist_threshold+1000)<t_tle.perigee_alt())==True):
            pyro.deterministic('conj',torch.tensor(False))
            return
        #If the above does not filter the solution, we might have a potential conjunction of the orbits:
        _,mu_earth,_,_,_,_,_,_=dsgp4.util.get_gravity_constants('wgs-84')
        self._mu_earth=float(mu_earth)*1e9
        # START SPACE GROUND TRUTH SIMULATION
        times = np.arange(self._time0, self._time0 + self._max_duration_days, self._delta_time)
        pyro.deterministic('time0',torch.tensor(self._time0))
        pyro.deterministic('max_duration_days', torch.tensor(self._max_duration_days))
        pyro.deterministic('delta_time', torch.tensor(self._delta_time))

        #Propagate target
        try:
            dsgp4.initialize_tle(t_tle)
            t_states = util.propagate_upsample(tle=t_tle, times_mjd=times, upsample_factor=self._time_upsample_factor)
            t_prop_error = torch.tensor(False)
        except:
            pyro.deterministic('conj',torch.tensor(False))
            t_prop_error = torch.tensor(True)
            return    
        pyro.deterministic('t_prop_error',t_prop_error)
        
        # Propagate chaser
        try:    
            dsgp4.initialize_tle(c_tle)
            c_states = util.propagate_upsample(tle=c_tle, times_mjd=times, upsample_factor=self._time_upsample_factor)
            c_prop_error = torch.tensor(False)
        except:
            pyro.deterministic('conj',torch.tensor(False))
            c_prop_error = torch.tensor(True)
            return
        pyro.deterministic('c_prop_error',c_prop_error)
        
        prop_error = c_prop_error or t_prop_error
        #pyro.deterministic('prop_error',prop_error)
        if prop_error:
            #pyro.deterministic('conj',torch.tensor(None))
            pyro.deterministic('conj',torch.tensor(False))
            return

        # Check for conjunction between target and chaser
        conj = torch.tensor(False)
        cdms = []
        t_positions = torch.from_numpy(np.array(t_states))[:,0]
        c_positions = torch.from_numpy(np.array(c_states))[:,0]
        i_min, d_min, i_conj, d_conj = find_conjunction(t_positions, c_positions, self._miss_dist_threshold)

        time_min = times[i_min]#self._time0 + i_min * self._delta_time
        time_conj = None
        if i_conj:
            conj = True
            time_conj = times[i_conj]#self._time0 + i_conj * self._delta_time    
            pyro.deterministic('time_min', torch.tensor(time_min))
            pyro.deterministic('d_min', d_min)
            pyro.deterministic('i_conj', torch.tensor(i_conj))
            pyro.deterministic('time_conj', torch.tensor(time_conj))
            pyro.deterministic('d_conj',d_conj)
        # END SPACE GROUND TRUTH SIMULATION


        # START GROUND SIMULATION
        mc_prop_error = False
        if conj:
            # Max number of CDMs to be issued
            max_ncdm = max(1, int((time_min - self._time0)*24.0/self._cdm_update_every_hours))
            pyro.deterministic('max_ncdm',torch.tensor(max_ncdm))#.tag(max_ncdm, 'max_ncdm')
            # We loop over the max number of CDMs and decide whether to issue CDMs according to Bernouilli distributions
            for i in range(max_ncdm):
                # We add some noise to the CDM issueing time:
                time_cdm = self._time0 + (i*self._cdm_update_every_hours)/24 #+ float(pyro.sample(fn=dist.Normal(loc=0, scale=0.001),name=f'noise_on_cdm_time_{i}'))
                i_cdm, time_cdm = util.find_closest(times, time_cdm)
                pyro.deterministic(f'time_cdm_{i}', torch.tensor(time_cdm))
                #pyro.deterministic(f'time_cdm', torch.tensor(time_cdm))
                # Tracking:
                # I store the time of new observation for target and chaser:
                # we ensure that at first both the objects are observed
                if i == 0:
                    t_new_obs = True
                    c_new_obs = True
                else:
                    # We sample the Bernouilli distribution to decide if there is a new observation:
                    t_new_obs = pyro.sample(f't_new_obs_{i}', Bernoulli(self._t_prob_new_obs))
                    c_new_obs = pyro.sample(f'c_new_obs_{i}', Bernoulli(self._c_prob_new_obs))
                # Tracking:
                t_state_new_obs, c_state_new_obs = None, None
                if t_new_obs:
                    ############## TODO: REVISIT THE WAY MULTIPLE OBSERVATIONS ARE COMBINED FOR AN OBJECT ##############
                    t_state_new_obs_list = []
                    t_cov_matrix_diagonal_obs_noise_list = []
                    for instrument in self._t_observing_instruments:
                        t_state_new_obs, t_cov_matrix_diagonal_obs_noise = instrument.observe(time_cdm, {'state_xyz': t_states[i_cdm]})
                        t_state_new_obs_list.append(t_state_new_obs)
                        t_cov_matrix_diagonal_obs_noise_list.append(t_cov_matrix_diagonal_obs_noise)

                    t_state_new_obs = np.stack(t_state_new_obs_list).mean(0)
                    self._t_cov_matrix_diagonal_obs_noise = np.stack(t_cov_matrix_diagonal_obs_noise_list).mean(0).squeeze()

                if c_new_obs:
                    ############## TODO: REVISIT THE WAY MULTIPLE OBSERVATIONS ARE COMBINED FOR AN OBJECT ##############
                    c_state_new_obs_list = []
                    c_cov_matrix_diagonal_obs_noise_list = []
                    for instrument in self._c_observing_instruments:
                        c_state_new_obs, c_cov_matrix_diagonal_obs_noise = instrument.observe(time_cdm, {'state_xyz': c_states[i_cdm]})
                        c_state_new_obs_list.append(c_state_new_obs)
                        c_cov_matrix_diagonal_obs_noise_list.append(c_cov_matrix_diagonal_obs_noise)

                    c_state_new_obs = np.stack(c_state_new_obs_list).mean(0)
                    self._c_cov_matrix_diagonal_obs_noise = np.stack(c_cov_matrix_diagonal_obs_noise_list).mean(0).squeeze()

                # We simulate CDM generation process and return CDM (is None if there are not new observation):
                # other_tle_info=[mean_motion_first_derivative, mean_motion_second_derivative, bstar]
                # print('Generate cdm')
                cdm = self.generate_cdm(t_state_new_obs=t_state_new_obs, c_state_new_obs=c_state_new_obs, time_obs_mjd=time_cdm,
                                time_conj_mjd=time_min,
                                t_tle=t_tle, c_tle=c_tle, previous_cdm=None if i==0 else cdms[-1])
                if mc_prop_error:
                    conj = False
                    cdms = []
                    break
                if cdm:
                    cdm_issued = True
                    cdms.append(cdm)
                else:
                    cdm_issued = False
                pyro. deterministic(f'cdm_issued_{i}', torch.tensor(cdm_issued))
        pyro.deterministic('mc_prop_error',torch.tensor(mc_prop_error))
        # END GROUND SIMULATION

        # Final things
        #pyro.deterministic('conj',torch.tensor(conj))
        prop_error = c_prop_error or t_prop_error or mc_prop_error
        pyro.deterministic('prop_error',torch.tensor(prop_error))
        num_cdms = len(cdms)
        pyro.deterministic('num_cdms', torch.tensor(num_cdms))
        pyro.sample("cdms", dist.Delta(torch.tensor(0.)), infer={"cdms": cdms})
        pyro.deterministic('conj',torch.tensor(conj))
        if conj:
            #let's store the two tles as well:
            pyro.sample("t_tle", dist.Delta(torch.tensor(0.)), infer={"t_tle": t_tle})
            pyro.sample("c_tle", dist.Delta(torch.tensor(0.)), infer={"c_tle": c_tle})

        #pyro.deterministic('cdms', torch.tensor(cdms))
        if conj:
            cdm0 = cdms[0]
            state0=cdm0.get_state(0)*1e3
            state1=cdm0.get_state(1)*1e3
            cdm0_t_sma, cdm0_t_ecc, cdm0_t_inc, _, _, _ = dsgp4.util.from_cartesian_to_keplerian(state0[0], state0[1], self._mu_earth)
            cdm0_c_sma, cdm0_c_ecc, cdm0_c_inc, _, _, _ = dsgp4.util.from_cartesian_to_keplerian(state1[0], state1[1], self._mu_earth)
            cdm0_time_to_tca = util.from_date_str_to_days(cdm0['TCA'], date0=cdm0['CREATION_DATE'])

            pyro.deterministic('cdm0_t_sma', torch.tensor(cdm0_t_sma))
            pyro.deterministic('cdm0_t_ecc', torch.tensor(cdm0_t_ecc))
            pyro.deterministic('cdm0_t_inc', torch.tensor(cdm0_t_inc))
            pyro.deterministic('cdm0_c_sma', torch.tensor(cdm0_c_sma))
            pyro.deterministic('cdm0_c_ecc', torch.tensor(cdm0_c_ecc))
            pyro.deterministic('cdm0_c_inc', torch.tensor(cdm0_c_inc))
            pyro.deterministic('cdm0_time_to_tca', torch.tensor(cdm0_time_to_tca))
            # Likelihood
            pyro.sample('obs_cdm0_t_sma', dist.Normal(cdm0_t_sma, self._likelihood_t_stddev[0]), obs=cdm0_t_sma)
            pyro.sample('obs_cdm0_t_ecc', dist.Normal(cdm0_t_ecc, self._likelihood_t_stddev[1]), obs=cdm0_t_ecc)
            pyro.sample('obs_cdm0_t_inc', dist.Normal(cdm0_t_inc, self._likelihood_t_stddev[2]), obs=cdm0_t_inc)
            pyro.sample('obs_cdm0_c_sma', dist.Normal(cdm0_c_sma, self._likelihood_c_stddev[0]), obs=cdm0_c_sma)
            pyro.sample('obs_cdm0_c_ecc', dist.Normal(cdm0_c_ecc, self._likelihood_c_stddev[1]), obs=cdm0_c_ecc)
            pyro.sample('obs_cdm0_c_inc', dist.Normal(cdm0_c_inc, self._likelihood_c_stddev[2]), obs=cdm0_c_inc)
            pyro.sample('obs_cdm0_time_to_tca', dist.Normal(cdm0_time_to_tca, self._likelihood_time_to_tca_stddev), obs=cdm0_time_to_tca)

    def get_conjunction(self):
        """
        This function generates a conjunction data message (``kessler.cdm.ConjunctionDataMessage``) from the current state of the chaser and target.
        It uses the ``self.forward()`` method to generate the conjunction and returns it.
        It returns the generated trace and the number of iterations it took to find a conjunction.

        ..note:: To extract the CDMs from the trace use the following code:
            >>> trace,it = model.get_conjunction()
            >>> cdms = trace.nodes['cdms']['infer']['cdms']
            >>> for cdm in cdms:
                    print(cdm)
        """
        found = False
        iteration = 0
        while not found:
            iteration += 1
            traced_model = pyro.poutine.trace(self.forward).get_trace()
            if traced_model.nodes['conj']['value']:
                found = True
        print(f"After {iteration} iterations, generated event with {len(traced_model.nodes['cdms']['infer']['cdms'])} CDMs")
        return traced_model, iteration
    

class ConjunctionSimplified(Conjunction):
    """

    This class is a simplified version of the Conjunction class. To generate the conjunction, it shuffles
    two TLEs from a given TLE population file, randomizing the mean anomaly, and checking for conjunctions.

    Example:
        >>> from kessler.model import ConjunctionSimplified
        >>> from dsgp4 import tle
        >>> tles = dsgp4.tle.load('tles_sample_population.txt')
        >>> model = ConjunctionSimplified(tles=tles)
        >>> tr = model.get_conjunction()
    """

    def __init__(self, tles, exclude_object_name=None, *args, **kwargs):
        self._tles = tles
        self._exclude_object_name=exclude_object_name
        if self._exclude_object_name is not None:
            include_idxs=[]
            for idx, tle in enumerate(self._tles):
                if tle.name.startswith(self._exclude_object_name)==False:
                    include_idxs.append(idx)
            self._include_idxs=include_idxs
        super().__init__(*args, **kwargs)

    def make_target(self):
        if self._t_tle is None:
            mean_anomaly = pyro.sample('t_mean_anomaly',self._prior_dict['mean_anomaly_prior'])
            tle_index = pyro.sample('t_sampled_index', Categorical(torch.tensor(range(len(self._tles)))))
            tle = self._tles[tle_index].copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyro.deterministic('t_mean_motion', torch.tensor(tle.mean_motion))
            pyro.deterministic('t_eccentricity', torch.tensor(tle.eccentricity))
            pyro.deterministic('t_inclination',torch.tensor(tle.inclination))
            pyro.deterministic('t_argument_of_perigee', torch.tensor(tle.argument_of_perigee))
            pyro.deterministic('t_raan',torch.tensor(tle.raan))
            pyro.deterministic('t_mean_motion_first_derivative',torch.tensor(tle.mean_motion_first_derivative))
            pyro.deterministic('t_mean_motion_second_derivative', torch.tensor(tle.mean_motion_second_derivative))
            pyro.deterministic('t_b_star',torch.tensor(tle.b_star))
        else:
            mean_anomaly = pyro.sample('t_mean_anomaly', self._prior_dict['mean_anomaly_prior'])
            tle = self._t_tle.copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyro.deterministic('t_mean_motion', torch.tensor(tle.mean_motion))
            pyro.deterministic('t_eccentricity', torch.tensor(tle.eccentricity))
            pyro.deterministic('t_inclination',torch.tensor(tle.inclination))
            pyro.deterministic('t_argument_of_perigee', torch.tensor(tle.argument_of_perigee))
            pyro.deterministic('t_raan',torch.tensor(tle.raan))
            pyro.deterministic('t_mean_motion_first_derivative',torch.tensor(tle.mean_motion_first_derivative))
            pyro.deterministic('t_mean_motion_second_derivative', torch.tensor(tle.mean_motion_second_derivative))
            pyro.deterministic('t_b_star',torch.tensor(tle.b_star))
        return tle

    def make_chaser(self):
        if self._c_tle is None:
            mean_anomaly = pyro.sample('c_mean_anomaly',self._prior_dict['mean_anomaly_prior'])
            if self._exclude_object_name is not None and self._t_tle_name.startswith(self._exclude_object_name):
                tle_index = pyro.sample('c_sampled_index',Categorical(torch.tensor(self._include_idxs)))
            else:
                tle_index = pyro.sample('c_sampled_index', Categorical(torch.tensor(range(len(self._tles)))))
            tle = self._tles[tle_index].copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyro.deterministic('c_mean_motion', torch.tensor(tle.mean_motion))
            pyro.deterministic('c_eccentricity', torch.tensor(tle.eccentricity))
            pyro.deterministic('c_inclination',torch.tensor(tle.inclination))
            pyro.deterministic('c_argument_of_perigee', torch.tensor(tle.argument_of_perigee))
            pyro.deterministic('c_raan',torch.tensor(tle.raan))
            pyro.deterministic('c_mean_motion_first_derivative',torch.tensor(tle.mean_motion_first_derivative))
            pyro.deterministic('c_mean_motion_second_derivative', torch.tensor(tle.mean_motion_second_derivative))
            pyro.deterministic('c_b_star',torch.tensor(tle.b_star))
        else:
            mean_anomaly = pyro.sample('c_mean_anomaly',self._prior_dict['mean_anomaly_prior'])
            tle = self._c_tle.copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyro.deterministic('c_mean_motion', torch.tensor(tle.mean_motion))
            pyro.deterministic('c_eccentricity', torch.tensor(tle.eccentricity))
            pyro.deterministic('c_inclination',torch.tensor(tle.inclination))
            pyro.deterministic('c_argument_of_perigee', torch.tensor(tle.argument_of_perigee))
            pyro.deterministic('c_raan',torch.tensor(tle.raan))
            pyro.deterministic('c_mean_motion_first_derivative',torch.tensor(tle.mean_motion_first_derivative))
            pyro.deterministic('c_mean_motion_second_derivative', torch.tensor(tle.mean_motion_second_derivative))
            pyro.deterministic('c_b_star',torch.tensor(tle.b_star))
        return tle