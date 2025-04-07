import numpy as np
import dsgp4
import torch
import uuid
# import warnings

from . import util, ConjunctionDataMessage
from dsgp4.tle import TLE

import pyprob
from pyprob import Model
from pyprob.distributions import Mixture, TruncatedNormal, Uniform, Normal, Bernoulli


def default_prior():
    """
    This function returns a dictionary of TLE elements priors, corresponding to the 22nd of
    May 2020. Each prior is a probability density function.

    Returns:
        - p (``dict``): dictionary of ``pyprob.distributions``
    """
    p={}
    p['mean_motion_prior']=Mixture(
        distributions=[TruncatedNormal(0.0010028142482042313, 0.00004670943133533001, low=0.0, high=0.004),
                       TruncatedNormal(0.00017592836171388626, 0.00003172305878251791, low=0.0, high=0.004),
                       TruncatedNormal(0.0010926761478185654, 0.000027726569678634405, low=0.0, high=0.004),
                       TruncatedNormal(0.0003353552892804146, 0.00007733114063739777, low=0.0, high=0.004),
                       TruncatedNormal(0.0007777251303195953, 0.00013636205345392227, low=0.0, high=0.004),
                       TruncatedNormal(0.001032940074801445, 0.00002651428570970893, low=0.0, high=0.004)],
        probs=[0.12375596165657043, 0.05202080309391022, 0.21220888197422028, 0.0373813770711422, 0.01674230769276619, 0.5578906536102295])
    p['mean_anomaly_prior'] = Uniform(low=0.0, high=6.2831854820251465)
    p['eccentricity_prior'] = Mixture(
        distributions=[TruncatedNormal(0.0028987403493374586, 0.002526970813050866, low=0.0, high=0.8999999761581421),
                       TruncatedNormal(0.6150050163269043, 0.07872536778450012, low=0.0, high=0.8999999761581421),
                       TruncatedNormal(0.05085373669862747, 0.024748045951128006, low=0.0, high=0.8999999761581421),
                       TruncatedNormal(0.3420163094997406, 0.18968918919563293, low=0.0, high=0.8999999761581421),
                       TruncatedNormal(0.7167646288871765, 0.011966796591877937, low=0.0, high=0.8999999761581421),
                       TruncatedNormal(0.013545362278819084, 0.0068586356937885284, low=0.0, high=0.8999999761581421)],
        probs=[0.5433819890022278, 0.04530993849039078, 0.08378008753061295, 0.02705608867108822, 0.03350389748811722, 0.2669680118560791])
    p['inclination_prior'] = Mixture(
        distributions=[TruncatedNormal(0.09954200685024261, 0.04205162078142166, low=0, high=3.1415),
                       TruncatedNormal(1.4393062591552734, 0.012214339338243008, low=0, high=3.1415),
                       TruncatedNormal(1.736578106880188, 0.11822951585054398, low=0, high=3.1415),
                       TruncatedNormal(1.0963480472564697, 0.010178830474615097, low=0, high=3.1415),
                       TruncatedNormal(0.48166394233703613, 0.04073172062635422, low=0, high=3.1415),
                       TruncatedNormal(0.9063634872436523, 0.04156989976763725, low=0, high=3.1415),
                       TruncatedNormal(1.275956392288208, 0.02754846028983593, low=0, high=3.1415),
                       TruncatedNormal(2.5208728313446045, 0.003279004478827119, low=0, high=3.1415),
                       TruncatedNormal(1.5189905166625977, 0.02461068518459797, low=0, high=3.1415),
                       TruncatedNormal(0.3474450707435608, 0.0433642603456974, low=0, high=3.1415),
                       TruncatedNormal(0.6648743152618408, 0.11472384631633759, low=0, high=3.1415),
                       TruncatedNormal(1.1465401649475098, 0.014345825649797916, low=0, high=3.1415),
                       TruncatedNormal(1.7207987308502197, 0.012212350033223629, low=0, high=3.1415)],
        probs=[0.028989605605602264, 0.10272273421287537, 0.02265254408121109, 0.019256576895713806, 0.028676774352788925, 0.06484941393136978, 0.13786117732524872, 0.0010146398562937975, 0.047179922461509705, 0.01607278548181057, 0.020023610442876816, 0.06644929945468903, 0.4442509114742279])
    p['argument_of_perigee_prior'] = Uniform(low=0.0, high=6.2829999923706055)
    p['raan_prior'] = Uniform(low=0.0, high=6.2829999923706055)
    p['mean_motion_first_derivative_prior'] = Normal(4.937096738377722e-13, 5.807570136601159e-13)
    p['b_star_prior'] = Mixture(
        distributions=[Normal(0.0002002030232688412, 0.0011279708705842495),
                       Normal(0.3039868175983429, 0.06403032690286636),
                       Normal(0.003936616238206625, 0.012939595617353916),
                       Normal(-0.04726288095116615, 0.17036935687065125),
                       Normal(0.08823495358228683, 0.061987943947315216)],
        probs=[0.9688150882720947, 0.0012630978599190712, 0.024090370163321495, 0.0009446446783840656, 0.0048867943696677685])
    return p


def find_conjunction(tr0, tr1, miss_dist_threshold):
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

class Conjunction(Model):
    """
    A class for simulating conjunctions.

    Args:
        time0 (``float``): The time of the first observation in MJD.
        max_duration_days (``float``): The maximum duration of the observations in days.
        time_resolution (``float``): The time resolution of the observations in seconds.
        time_upsample_factor (``int``): The factor by which the time resolution is upsampled.
        miss_dist_threshold (``float``): The miss distance threshold in meters.
        prior_dict (``dict``): A dictionary of priors for the parameters of the conjunctions. If None, a default dictionary is used.
        t_prob_new_obs (``float``): The probability of a new observation at the next CDM update.
        c_prob_new_obs (``float``): The probability of a new observation at the next CDM update.
        cdm_update_every_hours (``float``): The number of hours between consecutive CDM updates.
        mc_samples (``int``): The number of samples for Monte Carlo integration.
        mc_upsample_factor (``int``): The factor by which the time resolution is upsampled for Monte Carlo integration.
        pc_method (``str``): The method for computing the probability of collision. Can be 'MC' or 'CDM'.
        collision_threshold (``float``): The collision threshold in meters.
        likelihood_t_stddev (``list``): The standard deviation of the likelihood for the true anomaly
        likelihood_c_stddev (``list``): The standard deviation of the likelihood for the collision distance
        likelihood_time_to_tca_stdev (``list``): The standard deviation of the likelihood for the time to TCA.
    
    Returns:
        A model for conjunctions.
    """
    def __init__(self,
                 time0=58991.90384230018,
                 max_duration_days=7.0,
                 time_resolution=6e5,
                 time_upsample_factor=100,
                 miss_dist_threshold=20e3,
                 prior_dict=None,
                 t_prob_new_obs = 0.96,
                 c_prob_new_obs = 0.4,
                 cdm_update_every_hours = 8.,
                 mc_samples = 100,
                 mc_upsample_factor = 100,
                 pc_method = 'MC',
                 collision_threshold = 70,
                 likelihood_t_stddev = [3.71068006e+02, 9.99999999e-02, 1.72560879e-01],
                 likelihood_c_stddev = [3.71068006e+02, 9.99999999e-02, 1.72560879e-01],
                 likelihood_time_to_tca_stddev = 0.7,
                 t_observing_instruments = [],
                 c_observing_instruments = [],
                 t_tle = None,
                 c_tle = None
                ):

        self._time0 = time0
        self._max_duration_days = max_duration_days
        self._time_resolution = time_resolution
        self._time_upsample_factor = time_upsample_factor
        self._delta_time = max_duration_days / time_resolution
        self._miss_dist_threshold = miss_dist_threshold  # miss distance threshold in [m]
        if prior_dict is None:
            self._prior_dict = default_prior()
        else:
            self._prior_dict = prior_dict
        self._t_prob_new_obs = t_prob_new_obs
        self._c_prob_new_obs = c_prob_new_obs
        self._cdm_update_every_hours = cdm_update_every_hours
        self._mc_samples = mc_samples
        self._mc_upsample_factor = mc_upsample_factor
        self._pc_method = pc_method
        self._collision_threshold = collision_threshold
        self._likelihood_t_stddev = likelihood_t_stddev
        self._likelihood_c_stddev = likelihood_c_stddev
        self._likelihood_time_to_tca_stddev = likelihood_time_to_tca_stddev
        if len(t_observing_instruments)==0 or len(c_observing_instruments)==0:
            raise ValueError("We need at least one observing instrument for target and chaser!")
        self._t_observing_instruments = t_observing_instruments
        self._c_observing_instruments = c_observing_instruments
        self._t_tle = t_tle
        self._c_tle = c_tle
        super().__init__(name='Conjunction')

    def make_target(self):
        """"
        This function creates a target object, as a TLE (``dsgp4.tle.TLE``).
        """
        if self._t_tle is None:
            d = {}
            d['mean_motion'] = pyprob.sample(self._prior_dict['mean_motion_prior'], name='t_mean_motion')
            d['mean_anomaly'] = pyprob.sample(self._prior_dict['mean_anomaly_prior'], name='t_mean_anomaly')
            d['eccentricity'] = pyprob.sample(self._prior_dict['eccentricity_prior'], name='t_eccentricity')
            d['inclination'] = pyprob.sample(self._prior_dict['inclination_prior'], name='t_inclination')
            d['argument_of_perigee'] = pyprob.sample(self._prior_dict['argument_of_perigee_prior'], name='t_argument_of_perigee')
            d['raan'] = pyprob.sample(self._prior_dict['raan_prior'], name='t_raan')
            d['mean_motion_first_derivative'] = pyprob.sample(self._prior_dict['mean_motion_first_derivative_prior'], name='t_mean_motion_first_derivative')
            d['mean_motion_second_derivative'] = 0.0  # pybrob.sample(Uniform(0.0,1e-17))
            pyprob.tag(d['mean_motion_second_derivative'], 't_mean_motion_second_derivative')
            d['b_star'] = pyprob.sample(self._prior_dict['b_star_prior'], name='t_b_star')
            d['satellite_catalog_number'] = 43437
            d['classification'] = 'U'
            d['international_designator'] = '18100A'
            d['ephemeris_type'] = 0
            d['element_number'] = 9996
            d['revolution_number_at_epoch'] = 56353
            d['epoch_year'] = util.from_mjd_to_datetime(self._time0).year
            d['epoch_days'] = util.from_mjd_to_epoch_days_after_1_jan(self._time0)
            tle = TLE(d)
            return tle
        else:
            mean_anomaly = pyprob.sample(self._prior_dict['mean_anomaly_prior'], name = 't_mean_anomaly')
            tle = self._t_tle.copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyprob.tag(tle.mean_motion, name='t_mean_motion')
            pyprob.tag(tle.eccentricity, name='t_eccentricity')
            pyprob.tag(tle.inclination, name='t_inclination')
            pyprob.tag(tle.argument_of_perigee, name='t_argument_of_perigee')
            pyprob.tag(tle.raan, name='t_raan')
            pyprob.tag(tle.mean_motion_first_derivative, name='t_mean_motion_first_derivative')
            pyprob.tag(tle.mean_motion_second_derivative, name='t_mean_motion_second_derivative')
            pyprob.tag(tle.b_star, name='t_b_star')
            return tle

    def make_chaser(self):
        """
        This function creates a chaser object, as a TLE (``dsgp4.tle.TLE``).
        """
        if self._c_tle is None:
            d = {}
            d['mean_motion'] = pyprob.sample(self._prior_dict['mean_motion_prior'], name='c_mean_motion')
            d['mean_anomaly'] = pyprob.sample(self._prior_dict['mean_anomaly_prior'], name='c_mean_anomaly')
            d['eccentricity'] = pyprob.sample(self._prior_dict['eccentricity_prior'], name='c_eccentricity')
            d['inclination'] = pyprob.sample(self._prior_dict['inclination_prior'], name='c_inclination')
            d['argument_of_perigee'] = pyprob.sample(self._prior_dict['argument_of_perigee_prior'], name='c_argument_of_perigee')
            d['raan'] = pyprob.sample(self._prior_dict['raan_prior'], name='c_raan')
            d['mean_motion_first_derivative'] = pyprob.sample(self._prior_dict['mean_motion_first_derivative_prior'], name='c_mean_motion_first_derivative')
            d['mean_motion_second_derivative'] = 0.0  # pybrob.sample(Uniform(0.0,1e-17))
            pyprob.tag(d['mean_motion_second_derivative'], 'c_mean_motion_second_derivative')
            d['b_star'] = pyprob.sample(self._prior_dict['b_star_prior'], name='c_b_star')
            d['satellite_catalog_number'] = 43437
            d['classification'] = 'U'
            d['international_designator'] = '18100A'
            d['ephemeris_type'] = 0
            d['element_number'] = 9996
            d['revolution_number_at_epoch'] = 56353
            d['epoch_year'] = util.from_mjd_to_datetime(self._time0).year
            d['epoch_days'] = util.from_mjd_to_epoch_days_after_1_jan(self._time0)
            tle = TLE(d)
            return tle
        else:
            mean_anomaly = pyprob.sample(self._prior_dict['mean_anomaly_prior'], name = 'c_mean_anomaly')
            tle = self._c_tle.copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyprob.tag(tle.mean_motion, name='c_mean_motion')
            pyprob.tag(tle.eccentricity, name='c_eccentricity')
            pyprob.tag(tle.inclination, name='c_inclination')
            pyprob.tag(tle.argument_of_perigee, name='c_argument_of_perigee')
            pyprob.tag(tle.raan, name='c_raan')
            pyprob.tag(tle.mean_motion_first_derivative, name='c_mean_motion_first_derivative')
            pyprob.tag(tle.mean_motion_second_derivative, name='c_mean_motion_second_derivative')
            pyprob.tag(tle.b_star, name='c_b_star')
            return tle

    def generate_cdm(self, t_state_new_obs, c_state_new_obs, time_obs_mjd, time_conj_mjd, t_tle, c_tle, previous_cdm):
        """
        This function generates a conjunction data message (``kessler.cdm.ConjunctionDataMessage``) from the current state of the chaser and target.
        
        Args:
            t_state_new_obs (``torch.tensor``): The state of the target at the time of the observation.
            c_state_new_obs (``torch.tensor``): The state of the chaser at the time of the observation.
            time_obs_mjd (``float``): The time of the CDM in MJD.
            time_conj_mjd (``float``): The time of the conjunction in MJD.
            t_tle (``dsgp4.tle.TLE``): The TLE of the target.
            c_tle (``dsgp4.tle.TLE``): The TLE of the chaser.
            previous_cdm (``kessler.cdm.ConjunctionDataMessage``): The previous conjunction data message.
        
        Returns:
            cdm (``kessler.cdm.ConjunctionDataMessage``): The conjunction data message. None, if no CDM has to be generated.
        """
        # time_conj_mjd = time_conj_mjd + pyprob.sample(Normal(0, 0.00001))
        if c_state_new_obs is not None or t_state_new_obs is not None:
            # print('\n\n')
            # print('new cdm')
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
#                print("target")
#                print(t_tle,time_obs_mjd)
                # we return the state in XYZ, the state in RTN and the cov matrix in RTN
                if self._pc_method == 'MC':
                    obs_tle_t,_=dsgp4.newton_method(t_tle,time_obs_mjd)
                    t_mean_state_tca_xyz_TEME, t_cov_state_tca_rtn = self.propagate_uncertainty_monte_carlo(state_xyz = t_state_new_obs,
                                                                                                            cov_matrix_diagonal_rtn = self._t_cov_matrix_diagonal_obs_noise,
                                                                                                            time_obs_mjd = time_obs_mjd,
                                                                                                            time_ca_mjd = time_conj_mjd,
                                                                                                            obs_tle = obs_tle_t)
                    t_mean_state_tca_xyz_TEME = t_mean_state_tca_xyz_TEME / 1e3
                    t_mean_state_tca_xyz_ITRF = util.from_TEME_to_ITRF(t_mean_state_tca_xyz_TEME, tca_jd)
                    cdm.set_state(0, t_mean_state_tca_xyz_ITRF)
                    cdm.set_covariance(0, t_cov_state_tca_rtn)

            if c_state_new_obs is not None:
#                print("chaser")
#                print(c_tle,time_obs_mjd)
                obs_tle_c,_=dsgp4.newton_method(c_tle,time_obs_mjd)
                if self._pc_method == 'MC':
                    c_mean_state_tca_xyz_TEME, c_cov_state_tca_rtn = self.propagate_uncertainty_monte_carlo(state_xyz = c_state_new_obs,
                                                                                                            cov_matrix_diagonal_rtn = self._c_cov_matrix_diagonal_obs_noise,
                                                                                                            time_obs_mjd = time_obs_mjd,
                                                                                                            time_ca_mjd = time_conj_mjd,
                                                                                                            obs_tle = obs_tle_c)
                    c_mean_state_tca_xyz_TEME = c_mean_state_tca_xyz_TEME / 1e3
                    c_mean_state_tca_xyz_ITRF = util.from_TEME_to_ITRF(c_mean_state_tca_xyz_TEME, tca_jd)
                    cdm.set_state(1, c_mean_state_tca_xyz_ITRF)
                    cdm.set_covariance(1, c_cov_state_tca_rtn)
            #I now resample at TCA to find Pc:
            if self._pc_method == 'MC':
                #WARNING WARNING WARNING:
                #the following is a quick hack waiting for Pyprob to be able to sample w/o side effects
                rng_state = torch.random.get_rng_state()
                t_state_tca_rtn, _ = util.from_cartesian_to_rtn(cdm.get_state(0)*1e3)
                c_state_tca_rtn, _ = util.from_cartesian_to_rtn(cdm.get_state(1)*1e3)

                t_cov_pos_tca=cdm.get_covariance(0)[:3,:3]
                c_cov_pos_tca=cdm.get_covariance(1)[:3,:3]

                t_samples_tca=torch.distributions.MultivariateNormal(torch.tensor(t_state_tca_rtn[0]),torch.tensor(t_cov_pos_tca)).sample((int(self._mc_samples*self._mc_upsample_factor),))
                c_samples_tca=torch.distributions.MultivariateNormal(torch.tensor(c_state_tca_rtn[0]),torch.tensor(c_cov_pos_tca)).sample((int(self._mc_samples*self._mc_upsample_factor),))
                miss_distances=torch.cdist(t_samples_tca,c_samples_tca)
                #miss_distances=torch.norm(t_samples_tca-c_samples_tca,dim=-1)
                probability_of_collision=(miss_distances<self._collision_threshold).to(torch.int32).sum()/torch.numel(miss_distances)
#                print(f"probability of collision: {float(probability_of_collision)}")
#                print(f"minimum miss distance MC: {miss_distances.min()}")
#                print(f"mean miss distance MC: {miss_distances.mean()}")
                ########### TODO ###########
                #Recommended standard to compute Pc:
                #FOSTER-1992, CHAN-1997,PATERA-2001, and, ALFANO-2005
                #WARNING WARNING WARNING:
                #the following is a quick hack waiting for Pyprob to be able to sample w/o side effects
                torch.random.set_rng_state(rng_state)
                pyprob.tag(miss_distances.min(),"min_miss_dist_MC")

            pyprob.tag(probability_of_collision, "probability_of_collision")
            cdm.set_relative_metadata('COLLISION_PROBABILITY', float(probability_of_collision))
            cdm.set_relative_metadata('COLLISION_PROBABILITY_METHOD', self._pc_method)
            return cdm
        else:
            return None

    def propagate_uncertainty_monte_carlo(self, state_xyz, cov_matrix_diagonal_rtn, time_obs_mjd, time_ca_mjd, obs_tle):
        """
        This function propagates the uncertainty of the state of the target at the time of observation.

        Args:
            state_xyz (np.array): state of the target at the time of observation in cartesian coordinates
            cov_matrix_diagonal_rtn (np.array): diagonal of the covariance matrix of the state of the target at the time of observation in RTN coordinates
            time_obs_mjd (float): time of observation in MJD
            time_ca_mjd (float): time of closest approach in MJD
            obs_tle (dsgp4.TLE): TLE of the target at the time of observation

        Returns:
            mean_state_tca_xyz_TEME (np.array): mean state of the target at the time of closest approach in cartesian coordinates in TEME frame
            cov_state_tca_rtn (np.array): covariance matrix of the state of the target at the time of closest approach in RTN coordinates
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
        dtle_dx=util.keplerian_cartesian_partials(state_.requires_grad_(),self._mu_earth)
        #we construct the covariance matrix of the TLE elements via similarity transformation, using the partials of TLE elements w.r.t. cartesian coordinates
        Cov_tle=np.matmul(np.matmul(dtle_dx,C_xyz),dtle_dx.T)
        #we extract the mean TLE elements from the TLE object, that will be used for the sampling
        mean_tle_els=torch.tensor([obs_tle._no_kozai,obs_tle._ecco,obs_tle._inclo,obs_tle._nodeo,obs_tle._argpo,obs_tle._mo])
        #the below is another option, however, it breaks for circular and zero inclination orbits,
        #due to singularity on the elements
#        dx_dtle,y0=util.transformation_jacobian(obs_tle,0.)
#        Cov_tle=np.matmul(np.matmul(np.linalg.pinv(dx_tle),C_xyz),np.linalg.pinv(dx_tle.T))

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
        xpdotp   =  1440.0 / (2.0 *np.pi)
        no_kozai_conversion_factor=xpdotp/43200.0* np.pi

        ###### Covariance matrix sampling (directly in TLE elements) ######
        #WARNING WARNING WARNING:
        #the following is a quick hack waiting for Pyprob to be able to sample w/o side effects
        rng_state = torch.random.get_rng_state()
        try:
            dist=torch.distributions.MultivariateNormal(loc=mean_tle_els,covariance_matrix=torch.tensor(Cov_tle))
            samples=dist.sample((self._mc_samples,))
        except Exception as e:
            if "Expected parameter covariance_matrix" in str(e):
                if mean_tle_els[1]<0.:
                    tle_data['eccentricity']=torch.tensor(0.)#mean_tle_els[0]
                else:
                    tle_data['eccentricity']=mean_tle_els[1]
                tle_data['argument_of_perigee']=mean_tle_els[4]
                tle_data['inclination']=mean_tle_els[2]
                tle_data['mean_anomaly']=mean_tle_els[5]
                val=mean_tle_els[0]*no_kozai_conversion_factor
                #if 2*np.pi / val >= 225.0:
                #   tle_data['mean_motion']=(2*np.pi/(225.0*0.99))
                #else:
                tle_data['mean_motion']=val
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
        for sample in samples:
            if sample[1]<0.:
                tle_data['eccentricity']=torch.tensor(0.)
            else:
                tle_data['eccentricity']=sample[1]
            tle_data['argument_of_perigee']=sample[4]
            tle_data['inclination']=sample[2]
            tle_data['mean_anomaly']=sample[5]
            val=sample[0]*no_kozai_conversion_factor
#            if 2*np.pi / val >= 225.0:
#                tle_data['mean_motion']=(2*np.pi/(225.0*0.99))
#            else:
            tle_data['mean_motion']=val
            tle_data['raan']=sample[3]
            # Propagate object at tca
            tle_object = TLE(tle_data)
            dsgp4.initialize_tle(tle_object)
            tsinces=(torch.tensor(time_ca_mjd)-dsgp4.util.from_datetime_to_mjd(tle_object._epoch))*1440.
            mc_states_tca_xyz.append(dsgp4.propagate(tle_object,tsinces).numpy()*1e3)
        #WARNING WARNING WARNING:
        #the following is a quick hack waiting for Pyprob to be able to sample w/o side effects
        torch.random.set_rng_state(rng_state)
        mc_states_tca_xyz = np.stack(mc_states_tca_xyz)
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
        self._t_tle_name=t_tle.name
        c_tle = self.make_chaser()
        #we immediately exclude cases where target and chaser are the same object:
        if t_tle.name==c_tle.name:
            pyprob.tag(False, 'conj')
            return
        pyprob.tag(t_tle, 't_tle0')
        pyprob.tag(c_tle, 'c_tle0')
        #we use a perigee/apogee filter to immediately exclude cases
        if (((t_tle.apogee_alt()+self._miss_dist_threshold+1000)<c_tle.perigee_alt())==True) or (((c_tle.apogee_alt()+self._miss_dist_threshold+1000)<t_tle.perigee_alt())==True):
            pyprob.tag(False, 'conj')
            return
        #If the above does not filter the solution, we might have a potential conjunction of the orbits:
        _,mu_earth,_,_,_,_,_,_=dsgp4.util.get_gravity_constants('wgs-72')
        self._mu_earth=float(mu_earth)*1e9
        # START SPACE GROUND TRUTH SIMULATION
        times = np.arange(self._time0, self._time0 + self._max_duration_days, self._delta_time)
        pyprob.tag(self._time0, 'time0')
        pyprob.tag(self._max_duration_days, 'max_duration_days')
        pyprob.tag(self._delta_time, 'delta_time')

        #Propagate target
        dsgp4.initialize_tle(t_tle)
        try:
            t_states = util.propagate_upsample(tle=t_tle, times_mjd=times, upsample_factor=self._time_upsample_factor)
            t_prop_error = False
        except RuntimeError as e:
            # if str(e) == 'Error: Satellite decayed' or str(e) == 'Error: (e <= -0.001)' or str(e) == 'Eccentricity out of range' :
            #warnings.warn('Propagator error for target: {}'.format(str(e)))
            t_prop_error = True
        pyprob.tag(t_prop_error, 't_prop_error')

        # Propagate chaser
        dsgp4.initialize_tle(c_tle)
        try:
            c_states = util.propagate_upsample(tle=c_tle, times_mjd=times, upsample_factor=self._time_upsample_factor)
            c_prop_error = False
        except RuntimeError as e:
            #warnings.warn(f'Propagator error for chaser: {str(e)}')
            c_prop_error = True
        pyprob.tag(c_prop_error, 'c_prop_error')

        prop_error = c_prop_error or t_prop_error
        pyprob.tag(prop_error, 'prop_error')
        if prop_error:
            pyprob.tag(False, 'conj')
            return

        # Check for conjunction between target and chaser
        conj = False
        cdms = []
        t_positions = torch.from_numpy(np.array(t_states))[:,0]
        c_positions = torch.from_numpy(np.array(c_states))[:,0]
        i_min, d_min, i_conj, d_conj = find_conjunction(t_positions, c_positions, self._miss_dist_threshold)

        time_min = times[i_min]#self._time0 + i_min * self._delta_time
        time_conj = None
        if i_conj:
            conj = True
            time_conj = times[i_conj]#self._time0 + i_conj * self._delta_time

        pyprob.tag(time_min, 'time_min')
        pyprob.tag(d_min, 'd_min')

        pyprob.tag(i_conj, 'i_conj')
        pyprob.tag(time_conj, 'time_conj')
        # if conjunction:
        #     time_conj_likelihood = Normal(time_conj, 1)
        #     pyprob.observe(time_conj_likelihood, name='time_conj_obs')
        pyprob.tag(d_conj, 'd_conj')
        pyprob.tag(conj, 'conj')

        # END SPACE GROUND TRUTH SIMULATION


        # START GROUND SIMULATION
        mc_prop_error = False
        if conj:
            # Max number of CDMs to be issued
            max_ncdm = max(1, int((time_min - self._time0)*24.0/self._cdm_update_every_hours))
            pyprob.tag(max_ncdm, 'max_ncdm')
            # We loop over the max number of CDMs and decide whether to issue CDMs according to Bernouilli distributions
            for i in range(max_ncdm):
                # We add some noise to the CDM issueing time:
                time_cdm = self._time0 + (i*self._cdm_update_every_hours)/24 + float(pyprob.sample(Normal(0, 0.001)))
                i_cdm, time_cdm = util.find_closest(times, time_cdm)

                pyprob.tag(time_cdm, f'time_cdm_{i}')
                pyprob.tag(time_cdm, 'time_cdm')
                # Tracking:
                # I store the time of new observation for target and chaser:
                # we ensure that at first both the objects are observed
                if i == 0:
                    t_new_obs = True
                    c_new_obs = True
                else:
                    t_new_obs = pyprob.sample(Bernoulli(self._t_prob_new_obs), name=f't_new_obs_{i}')
                    c_new_obs = pyprob.sample(Bernoulli(self._c_prob_new_obs), name=f'c_new_obs_{i}')

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
                pyprob.tag(cdm_issued, f'cdm_issued_{i}')
        pyprob.tag(mc_prop_error, 'mc_prop_error')
        # END GROUND SIMULATION

        # Final things
        pyprob.tag(conj, name='conj')
        prop_error = c_prop_error or t_prop_error or mc_prop_error
        pyprob.tag(prop_error, 'prop_error')
        num_cdms = len(cdms)
        pyprob.tag(num_cdms, 'num_cdms')
        pyprob.tag(cdms, 'cdms')

        # Likelihood model
        if conj:
            cdm0 = cdms[0]
            cdm0_t_sma, cdm0_t_ecc, cdm0_t_inc, _, _, _ = dsgp4.util.from_cartesian_to_keplerian(cdm0.get_state(0)*1e3, self._mu_earth)
            cdm0_c_sma, cdm0_c_ecc, cdm0_c_inc, _, _, _ = dsgp4.util.from_cartesian_to_keplerian(cdm0.get_state(1)*1e3, self._mu_earth)
            cdm0_time_to_tca = util.from_date_str_to_days(cdm0['TCA'], date0=cdm0['CREATION_DATE'])

            pyprob.tag(cdm0_t_sma, name='cdm0_t_sma')
            pyprob.tag(cdm0_t_ecc, name='cdm0_t_ecc')
            pyprob.tag(cdm0_t_inc, name='cdm0_t_inc')
            pyprob.tag(cdm0_c_sma, name='cdm0_c_sma')
            pyprob.tag(cdm0_c_ecc, name='cdm0_c_ecc')
            pyprob.tag(cdm0_c_inc, name='cdm0_c_inc')
            pyprob.tag(cdm0_time_to_tca, name='cdm0_time_to_tca')

            pyprob.observe(Normal(cdm0_t_sma, self._likelihood_t_stddev[0]), name='obs_cdm0_t_sma')
            pyprob.observe(Normal(cdm0_t_ecc, self._likelihood_t_stddev[1]), name='obs_cdm0_t_ecc')
            pyprob.observe(Normal(cdm0_t_inc, self._likelihood_t_stddev[2]), name='obs_cdm0_t_inc')
            pyprob.observe(Normal(cdm0_c_sma, self._likelihood_c_stddev[0]), name='obs_cdm0_c_sma')
            pyprob.observe(Normal(cdm0_c_ecc, self._likelihood_c_stddev[1]), name='obs_cdm0_c_ecc')
            pyprob.observe(Normal(cdm0_c_inc, self._likelihood_c_stddev[2]), name='obs_cdm0_c_inc')
            pyprob.observe(Normal(cdm0_time_to_tca, self._likelihood_time_to_tca_stddev), name='obs_cdm0_time_to_tca')

    def get_conjunction(self):
        found = False
        iter=0
        while not found:
            iter+=1
            trace = self.get_trace()
            if trace['conj']:
                found = True
        print(f"After {iter} iterations, generated event with {len(trace['cdms'])} CDMs")
        return trace,iter

from pyprob.distributions import Categorical

class ConjunctionSimplified(Conjunction):
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
            mean_anomaly = pyprob.sample(self._prior_dict['mean_anomaly_prior'], name = 't_mean_anomaly')
            tle_index = pyprob.sample( Categorical(range(len(self._tles))) , name='t_sampled_index')
            tle = self._tles[tle_index].copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyprob.tag(tle.mean_motion, name='t_mean_motion')
            pyprob.tag(tle.eccentricity, name='t_eccentricity')
            pyprob.tag(tle.inclination, name='t_inclination')
            pyprob.tag(tle.argument_of_perigee, name='t_argument_of_perigee')
            pyprob.tag(tle.raan, name='t_raan')
            pyprob.tag(tle.mean_motion_first_derivative, name='t_mean_motion_first_derivative')
            pyprob.tag(tle.mean_motion_second_derivative, name='t_mean_motion_second_derivative')
            pyprob.tag(tle.b_star, name='t_b_star')
        else:
            mean_anomaly = pyprob.sample(self._prior_dict['mean_anomaly_prior'], name = 't_mean_anomaly')
            tle = self._t_tle.copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyprob.tag(tle.mean_motion, name='t_mean_motion')
            pyprob.tag(tle.eccentricity, name='t_eccentricity')
            pyprob.tag(tle.inclination, name='t_inclination')
            pyprob.tag(tle.argument_of_perigee, name='t_argument_of_perigee')
            pyprob.tag(tle.raan, name='t_raan')
            pyprob.tag(tle.mean_motion_first_derivative, name='t_mean_motion_first_derivative')
            pyprob.tag(tle.mean_motion_second_derivative, name='t_mean_motion_second_derivative')
            pyprob.tag(tle.b_star, name='t_b_star')
        return tle

    def make_chaser(self):
        if self._c_tle is None:
            mean_anomaly = pyprob.sample(self._prior_dict['mean_anomaly_prior'], name = 'c_mean_anomaly')
            if self._exclude_object_name is not None and self._t_tle_name.startswith(self._exclude_object_name):
                tle_index = pyprob.sample( Categorical(self._include_idxs) , name='c_sampled_index')
            else:
                tle_index = pyprob.sample( Categorical(range(len(self._tles))) , name='c_sampled_index')
            tle = self._tles[tle_index].copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyprob.tag(tle.mean_motion, name='c_mean_motion')
            pyprob.tag(tle.eccentricity, name='c_eccentricity')
            pyprob.tag(tle.inclination, name='c_inclination')
            pyprob.tag(tle.argument_of_perigee, name='c_argument_of_perigee')
            pyprob.tag(tle.raan, name='c_raan')
            pyprob.tag(tle.mean_motion_first_derivative, name='c_mean_motion_first_derivative')
            pyprob.tag(tle.mean_motion_second_derivative, name='c_mean_motion_second_derivative')
            pyprob.tag(tle.b_star, name='c_b_star')
        else:
            mean_anomaly = pyprob.sample(self._prior_dict['mean_anomaly_prior'], name = 'c_mean_anomaly')
            tle = self._c_tle.copy()
            tle.update({'mean_anomaly': mean_anomaly})
            pyprob.tag(tle.mean_motion, name='c_mean_motion')
            pyprob.tag(tle.eccentricity, name='c_eccentricity')
            pyprob.tag(tle.inclination, name='c_inclination')
            pyprob.tag(tle.argument_of_perigee, name='c_argument_of_perigee')
            pyprob.tag(tle.raan, name='c_raan')
            pyprob.tag(tle.mean_motion_first_derivative, name='c_mean_motion_first_derivative')
            pyprob.tag(tle.mean_motion_second_derivative, name='c_mean_motion_second_derivative')
            pyprob.tag(tle.b_star, name='c_b_star')
        return tle
