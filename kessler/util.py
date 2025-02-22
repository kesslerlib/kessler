# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
# Trillium Technologies
# Giacomo Acciarini
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.


import numpy as np
import torch
import math
import os
import sys
import time
import datetime
import functools
import random
import dsgp4
import matplotlib.pyplot as plt


_print_refresh_rate = 0.25  #


def seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    global _random_seed
    _random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed()

def fit_mixture(values, *args, **kwargs):
    """
    This function fits a mixture of Gaussians to the provided values.

    Args:
        values (`numpy.ndarray`): values to fit the mixture to

    Returns:
        tuple: tuple containing: - `numpy.ndarray`: means of the mixture - `numpy.ndarray`: standard deviations of the mixture - `numpy.ndarray`: weights of the mixture
    """
    from sklearn import mixture
    values = values.reshape(-1,1)
    m = mixture.GaussianMixture(*args, **kwargs)
    m.fit(values)
    return m

def keplerian_cartesian_partials(state,mu):
    """
    Computes the partial derivatives of the cartesian state with respect to the keplerian elements.

    Args:
        state (`numpy.array`): numpy array of 2 rows and 3 columns, where
                                    the first row represents position, and the second velocity.
        mu (`float`): gravitational parameter of the central body

    Returns:
        `numpy.array`: numpy array of the partial derivatives of the cartesian state with respect to the keplerian elements.
    """
    state_1=dsgp4.util.clone_w_grad(state)
    state_2=dsgp4.util.clone_w_grad(state)
    state_3=dsgp4.util.clone_w_grad(state)
    state_4=dsgp4.util.clone_w_grad(state)
    state_5=dsgp4.util.clone_w_grad(state)
    a=dsgp4.util.from_cartesian_to_keplerian_torch(state,mu=mu)[0]
    a.backward()
    gradient_a=state.grad.flatten()
    e=dsgp4.util.from_cartesian_to_keplerian_torch(state_1,mu=mu)[1]
    e.backward()
    gradient_e=state_1.grad.flatten()
    i=dsgp4.util.from_cartesian_to_keplerian_torch(state_2,mu=mu)[2]
    i.backward()
    gradient_i=state_2.grad.flatten()
    Omega=dsgp4.util.from_cartesian_to_keplerian_torch(state_3,mu=mu)[3]
    Omega.backward()
    gradient_Omega=state_3.grad.flatten()
    omega=dsgp4.util.from_cartesian_to_keplerian_torch(state_4,mu=mu)[4]
    omega.backward()
    gradient_omega=state_4.grad.flatten()
    mean_anomaly=dsgp4.util.from_cartesian_to_keplerian_torch(state_5,mu=mu)[5]
    mean_anomaly.backward()
    gradient_mean_anomaly=state_5.grad.flatten()
    DF=np.stack((gradient_a, gradient_e, gradient_i, gradient_Omega, gradient_omega, gradient_mean_anomaly))
    return DF

def find_closest(values, t):
    indx = np.argmin(abs(values-t))
    return indx, values[indx]


def upsample(s, target_resolution):
    s = s.transpose(0, 1)
    s = torch.nn.functional.interpolate(s.unsqueeze(0), size=(target_resolution), mode='linear', align_corners=True)
    s = s.squeeze(0).transpose(0, 1)
    return s


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def rotation_matrix(state):
    """
    Computes the UVW rotation matrix.

    Args:
        state (`numpy.array`): numpy array of 2 rows and 3 columns, where
                                    the first row represents position, and the second velocity.

    Returns:
        `numpy.array`: numpy array of the rotation matrix from the cartesian state.
    """
    r, v = state[0], state[1]
    u = r / np.linalg.norm(r)
    w = np.cross(r, v)
    w = w / np.linalg.norm(w)
    v = np.cross(w, u)
    return np.vstack((u, v, w))


def from_cartesian_to_rtn(state, cartesian_to_rtn_rotation_matrix=None):
    """
    Converts a cartesian state to the RTN frame.

    Args:
        state (`numpy.array`): numpy array of 2 rows and 3 columns, where
                                    the first row represents position, and the second velocity.
        cartesian_to_rtn_rotation_matrix (`numpy.array`): numpy array of the rotation matrix from the cartesian state. If None, it is computed.

    Returns:
        `numpy.array`: numpy array of the RTN state.
    """
    # Use the supplied rotation matrix if available, otherwise compute it
    if cartesian_to_rtn_rotation_matrix is None:
        cartesian_to_rtn_rotation_matrix = rotation_matrix(state)
    r, v = state[0], state[1]
    r_rtn = np.dot(cartesian_to_rtn_rotation_matrix, r)
    v_rtn = np.dot(cartesian_to_rtn_rotation_matrix, v)
    return np.stack([r_rtn, v_rtn]), cartesian_to_rtn_rotation_matrix


def from_rtn_to_cartesian(state_rtn, rtn_to_cartesian_rotation_matrix):
    """
    Converts a RTN state to the cartesian frame.

    Args:
        state_rtn (`numpy.array`): numpy array of 2 rows and 3 columns, where
                                    the first row represents position, and the second velocity.
        rtn_to_cartesian_rotation_matrix (`numpy.array`): numpy array of the rotation matrix from the RTN state.

    Returns:
        `numpy.array`: numpy array of the cartesian state.
    """
    r_rtn, v_rtn = state_rtn[0], state_rtn[1]
    state_xyz = np.stack([np.matmul(rtn_to_cartesian_rotation_matrix, r_rtn), np.matmul(rtn_to_cartesian_rotation_matrix, v_rtn)])
    return state_xyz


def from_TEME_to_ITRF(state, time):
    import skyfield.sgp4lib
    """
    This function transforms the state from TEME reference frame, to ITRF reference frame, at a given time.
    TEME (True Equator, Mean Equinox reference frame)--> Earth-Centered Inertial Reference Frame
    ITRF (International Terrestrial Reference Frame)--> Earth-Centered Earth-Fixed Reference Frame

    Args:
        state (`numpy.array`): a numpy array of two elements, where each element is a three-dimensional array (position and velocity)
                             expressed in TEME reference frame
        time (`float`): time at which the conversion has to happen (as a Julian date)

    Returns:
        `numpy.array`: the transformed state (same dimensions as the input state)

    .. note::
        This method relies on a 3rd party library for the conversion (Skyfield:
        https://github.com/skyfielders/python-skyfield).
    """
    r, v = state[0], state[1]
    # time must be in J2000
    # velocity in the converter is in m/days, so we multiply by 86400 before conversion and divide later
    # print(f'pos: {r}, vel: {v}')
    r_new, v_new = skyfield.sgp4lib.TEME_to_ITRF(time, r, v*86400.)
    # print(f'pos: {r_new}, vel: {v_new/86400.}')
    v_new = v_new / 86400.
    state = np.stack([r_new, v_new])
    return state

def from_datetime_to_fractional_day(datetime_object):
    """
    Converts a datetime object to a fractional day. The fractional day is the number of days since the beginning of the year. For example, January 1st is 0.0, January 2nd is 1.0, etc.

    Args:
        datetime_object (`datetime.datetime`): datetime object to convert

    Returns:
        `float`: fractional day
    """
    d = datetime_object-datetime.datetime(datetime_object.year-1, 12, 31)
    fractional_day = d.days + d.seconds/60./60./24 + d.microseconds/60./60./24./1e6
    return fractional_day

def from_datetime_to_cdm_datetime_str(datetime):
    return datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')

def from_mjd_to_jd(mjd_date):
    """
    Converts a Modified Julian Date to a Julian Date. The Julian Date is the number of days since noon on January 1st, 4713 BC. The Modified Julian Date is the number of days since midnight on November 17th, 1858.

    Args:
        mjd_date (`float`): Modified Julian Date

    Returns:
        `float`: Julian Date
    """
    return mjd_date+2400000.5

def from_jd_to_mjd(jd_date):
    """
    Converts a Julian Date to a Modified Julian Date. The Julian Date is the number of days since noon on January 1st, 4713 BC.

    Args:
        jd_date (`float`): Julian Date

    Returns:
        `float`: Modified Julian Date
    """
    return jd_date-2400000.5

def from_jd_to_cdm_datetime_str(jd_date):
    d = dsgp4.util.from_jd_to_datetime(jd_date)
    return from_datetime_to_cdm_datetime_str(d)


def from_mjd_to_epoch_days_after_1_jan(mjd_date):
    d = dsgp4.util.from_mjd_to_datetime(mjd_date)
    dd = d - datetime.datetime(d.year, 1, 1)
    days = dd.days
    days_fraction = (dd.seconds + dd.microseconds/1e6) / (60*60*24)
    return days + days_fraction

def from_mjd_to_datetime_offset_aware(mjd_date):
    """
    Converts a Modified Julian Date to a datetime object. The Modified Julian Date is the number of days since midnight on November 17, 1858.

    Args:
        mjd_date (`float`): Modified Julian Date

    Returns:
        `datetime.datetime`: datetime object
    """
    datetime_obj=dsgp4.util.from_mjd_to_datetime(mjd_date)
    return datetime_obj.replace(tzinfo = datetime.timezone.utc)

def from_string_to_datetime(string):
    """
    Converts a string to a datetime object.

    Args:
        string (`str`): string to convert

    Returns:
        `datetime.datetime`: datetime object
    """
    if string.find('.')!=-1:
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S.%f')
    else:
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')


@functools.lru_cache(maxsize=None)
def from_date_str_to_days(date, date0='2020-05-22T21:41:31.975', date_format='%Y-%m-%dT%H:%M:%S.%f'):
    date = datetime.datetime.strptime(date, date_format)
    date0 = datetime.datetime.strptime(date0, date_format)
    dd = date-date0
    days = dd.days
    days_fraction = (dd.seconds + dd.microseconds/1e6) / (60*60*24)
    return days + days_fraction


def add_days_to_date_str(date0, days):
    date0 = datetime.datetime.strptime(date0, '%Y-%m-%dT%H:%M:%S.%f')
    date = date0 + datetime.timedelta(days=days)
    return from_datetime_to_cdm_datetime_str(date)


def is_date(date_string, date_format):
    try:
        datetime.datetime.strptime(date_string, date_format)
        return True
    except:
        return False


def transform_date_str(date_string, date_format_from, date_format_to):
    date = datetime.datetime.strptime(date_string, date_format_from)
    return date.strftime(date_format_to)


def find_closest(values, t):
    """
    Finds the closest value in a list of values to a given value.

    Args:
        values (`list`): list of values
        t (`float`): value to find the closest to

    Returns:
        `float`: closest value in the list to the given value
    """
    indx = np.argmin(abs(values-t))
    return indx, values[indx]

def upsample(s, target_resolution):
    """
    Upsamples a tensor to a given resolution, via linear interpolation.

    Args:
        s (`torch.Tensor`): tensor to upsample
        target_resolution (`int`): target resolution

    Returns:
        `torch.Tensor`: upsampled tensor
    """
    s = s.transpose(0, 1)
    s = torch.nn.functional.interpolate(s.unsqueeze(0), size=(target_resolution), mode='linear', align_corners=True)
    s = s.squeeze(0).transpose(0, 1)
    return s

def create_path(path, directory=False):
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print('{} does not exist, creating'.format(dir))
        try:
            os.makedirs(dir)
        except Exception as e:
            print(e)
            print('Could not create path, potentially created by another process in the meantime: {}'.format(path))


def tile_rows_cols(num_items):
    if num_items < 5:
        return 1, num_items
    else:
        cols = math.ceil(math.sqrt(num_items))
        rows = 0
        while num_items > 0:
            rows += 1
            num_items -= cols
        return rows, cols


def has_nan_or_inf(value):
    if torch.is_tensor(value):
        value = torch.sum(value)
        isnan = int(torch.isnan(value)) > 0
        isinf = int(torch.isinf(value)) > 0
        return isnan or isinf
    else:
        value = float(value)
        return math.isnan(value) or math.isinf(value)


def trace_to_event(trace):
    from .event import Event
    return Event(cdms=trace['cdms'])


def dist_to_event_dataset(dist):
    from .event import EventDataset
    return EventDataset(events=list(map(trace_to_event, dist)))


def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))


def progress_bar(i, len):
    bar_len = 20
    filled_len = int(round(bar_len * i / len))
    # percents = round(100.0 * i / len, 1)
    return '#' * filled_len + '-' * (bar_len - filled_len)


progress_bar_num_iters = None
progress_bar_len_str_num_iters = None
progress_bar_time_start = None
progress_bar_prev_duration = None


def progress_bar_init(message, num_iters, iter_name='Items'):
    global progress_bar_num_iters
    global progress_bar_len_str_num_iters
    global progress_bar_time_start
    global progress_bar_prev_duration
    if num_iters < 0:
        raise ValueError('num_iters must be a non-negative integer')
    progress_bar_num_iters = num_iters
    progress_bar_time_start = time.time()
    progress_bar_prev_duration = 0
    progress_bar_len_str_num_iters = len(str(progress_bar_num_iters))
    print(message)
    sys.stdout.flush()
    if progress_bar_num_iters > 0:
        print('Time spent  | Time remain.| Progress             | {} | {}/sec'.format(iter_name.ljust(progress_bar_len_str_num_iters * 2 + 1), iter_name))


def progress_bar_update(iter):
    global progress_bar_prev_duration
    if progress_bar_num_iters > 0:
        duration = time.time() - progress_bar_time_start
        if (duration - progress_bar_prev_duration > _print_refresh_rate) or (iter >= progress_bar_num_iters - 1):
            progress_bar_prev_duration = duration
            traces_per_second = (iter + 1) / duration
            print('{} | {} | {} | {}/{} | {:,.2f}       '.format(days_hours_mins_secs_str(duration), days_hours_mins_secs_str((progress_bar_num_iters - iter) / traces_per_second), progress_bar(iter, progress_bar_num_iters), str(iter).rjust(progress_bar_len_str_num_iters), progress_bar_num_iters, traces_per_second), end='\r')
            sys.stdout.flush()


def progress_bar_end(message=None):
    progress_bar_update(progress_bar_num_iters)
    print()
    if message is not None:
        print(message)

def get_ccsds_time_format(time_string):
    """
    Adapted by Andrew Ng, 18/3/2022.  
    Original MATLAB source code:  
    `NASA CARA Analysis Tools <https://github.com/nasa/CARA_Analysis_Tools>`_  

    Processes and outputs the format of the time string extracted from the CDM.  
    The CCSDS time format is required to be of the general form:  

    .. code-block:: none

        yyyy-[mm-dd|ddd]THH:MM:SS[.F*][Z]

    Format Rules:
        1. The date and time fields are separated by a **"T"**.
        2. The date field consists of a **four-digit year**, followed by either:
            - A two-digit month and a two-digit day, or  
            - A three-digit day-of-year.
        3. The year, month, day, and day-of-year fields are separated by a **dash ("-")**.
        4. The hours, minutes, and seconds fields are **two-digit values** separated by **colons (":")**.
        5. The fraction of seconds is optional and can have **any number of digits**.
        6. If a fraction of seconds is provided, it is separated from the two-digit seconds by a **period (".")**.
        7. The time string can end with an optional **"Z"** time zone indicator.

    Args:
        time_string (str): Original time string stored in CDM.

    Returns:
        str: Outputs the format of the time string.  
        Must be of the form **yyyy-[mm-dd|ddd]THH:MM:SS[.F*][Z]**, otherwise a `RuntimeError` is raised.
    """

    time_format = []
    numT = time_string.count('T')
    if numT == -1:
        # Case when this is 'T' does not exist in time_string
        raise RuntimeError(f"*** Error -- Invalid CCSDS time string: {time_string}\nNo 'T' separator found between date and time portions of the string")
    elif numT > 1:
        raise RuntimeError(f"*** Error -- Invalid CCSDS time string: {time_string} \nMore than one 'T' separator found between date and time portions of the string")
    idx_T = time_string.find('T')
    if idx_T ==10:
        time_format = "yyyy-mm-ddTHH:MM:SS"
    elif idx_T ==8:
        time_format = "yyyy-DDDTHH:MM:SS"
    else: 
        raise RuntimeError(f"*** Error -- Invalid CCSDS time string: {time_string} \nDate format not one of yyyy-mm-dd or yyyy-DDD.\n")
    # % Check if 'Z' time zone indicator appended to the string
    if time_string[-1]=='Z':
        z_opt = True
    else:
        z_opt = False
    # % Find location of the fraction of seconds decimal separator
    num_decimal = time_string.count('.')
    if num_decimal > 1:
        #time_format = []
        raise RuntimeError(f"*** Error -- Invalid CCSDS time string: {time_string}\nMore than one fraction of seconds decimal separator ('.') found.\n")
    idx_decimal = time_string.find('.')
    nfrac = 0
    if num_decimal != 0:
        if z_opt:
            nfrac = len(time_string) - 1 - idx_decimal -1
        else: 
            nfrac = len(time_string) - 1 - idx_decimal
    if nfrac > 0:
        frac_str = '.' + ('F'*nfrac)
    else:
        frac_str = ""
    if z_opt:
        frac_str = frac_str+'Z'
    time_format = time_format + frac_str
    return time_format

def doy_2_date(value, doy, year, idx):
    '''
    Written by Andrew Ng, 18/03/2022, 
    Based on source code @ https://github.com/nasa/CARA_Analysis_Tools/blob/master/two-dimension_Pc/Main/TransformationCode/TimeTransformations/DOY2Date.m
    Use the datetime python package. 
    doy_2_date  - Converts Day of Year (DOY) date format to date format.
    
    Args:
        - value(``str``): Original date time string with day of year format "YYYY-DDDTHH:MM:SS.ff"
        - doy  (``str``): The day of year in the DOY format. 
        - year (``str``): The year.
        - idx  (``int``): Index of the start of the original "value" string at which characters 'DDD' are found. 
    Returns: 
        -value (``str``): Transformed date in traditional date format. i.e.: "YYYY-mm-ddTHH:MM:SS.ff"

    '''
    # Calculate datetime format
    date_num = datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(doy) - 1)

    # Split datetime object into a date list
    date_vec = [date_num.year, date_num.month, date_num.day, date_num.hour, date_num.minute]
    # Extract final date string. Use zfill() to pad year, month and day fields with zeroes if not filling up sufficient spaces. 
    value = str(date_vec[0]).zfill(4) +'-' + str(date_vec[1]).zfill(2) + '-' + str(date_vec[2]).zfill(2) + 'T' + value[idx+4:-1] 
    return value

### Megaconstellation Builder ###
def build_megaconstellation(launch_date,
                            constellation_name='starlink',
                            groups='all',
                            mu_earth=398600800000000.0):
    """
    This function creates a list of TLEs for a given specified megaconstellation.
    It currently supports: starlink (in particular: Phase-1, as of 2022/01, all groups),
    oneweb (in particular, Phase 2 as of 2020/05, all groups), Amazon Kuiper, Guo Wang, and SatRevolution. 
    The user, can specify
    the launch date, the constellation name, which group to be selected, and the gravitational
    parameter to be used, and a list of TLEs corresponding to the specified constellation is
    returned.

    Args:
        launch_date (`datetime.datetime`): launch date as a datetime object
        constellation_name (`str`)
        groups (`str` or `int`): group number as an integer, or 'all' in case all groups shall be selected
        mu_earth (`float`): gravitational parameter of the Earth i m^3/s^2

    Returns:
        `list`: list of TLE (`dsgp4.tle.TLE`) objects
    """
    from . import TLE

    if constellation_name not in ['starlink','oneweb','kuiper','guowang','satrevolution']:
        raise ValueError(f"Only starlink, oneweb, kuiper, guowang, and satrevolution are supported, whereas {constellation_name} provided")
    if isinstance(groups,str):
        if groups!='all':
            raise ValueError(f"Only acceptable value for groups as a string is 'all', whereas {groups} provided")
    if isinstance(groups,int):
        if constellation_name=='starlink':
            if groups not in [1,2,3,4,5]:
                raise ValueError(f"Only group values of: 1 or 2 or 3 or 4 or 5 are valid.; while {groups} provided")
        elif constellation_name=='oneweb':
            if groups not in [1,2,3,4]:
                raise ValueError(f"Only group values of: 1 or 2 or 3 or 4 are valid; while {groups} provided")
        elif constellation_name=='kuiper':
            if groups not in [1,2,3]:
                raise ValueError(f"Only group values of: 1 or 2 or 3 are valid; while {groups} provided")
        elif constellation_name=='guowang':
            if groups not in [1,2,3,4,5,6,7]:
                raise ValueError(f"Only group values of: 1 or 2 or 3 or 4 are valid; while {groups} provided")
        elif constellation_name=='satrevolution':
            if groups not in [1,2]:
                raise ValueError(f"Only group values of: 1 or 2 are valid; while {groups} provided")
    if isinstance(launch_date,float):
        launch_date=from_mjd_to_datetime(launch_date)
    print(f"Launch date: {launch_date}, for constellation: {constellation_name}, group: {groups}")
    epoch_year=launch_date.year
    epoch_days=from_datetime_to_fractional_day(launch_date)
    tles=[]

    if constellation_name=='starlink':
        starlink_dic={"group_1":
                                   {"inclination":np.deg2rad(53),
                                    "planes":72,
                                    "number":1584,
                                    "altitude":6371*1e3+550*1e3},
                        "group_2":
                                  {"inclination":np.deg2rad(70),
                                    "planes":36,
                                    "number":720,
                                    "altitude":6371*1e3+570*1e3
                                  },
                        "group_3":
                                  {"inclination":np.deg2rad(97.6),
                                    "planes":6,
                                    "number":348,
                                    "altitude":6371*1e3+560*1e3
                                  },
                        "group_4":
                                  {"inclination":np.deg2rad(53.2),
                                    "planes":72,
                                    "number":1584,
                                    "altitude":6371*1e3+540*1e3
                                  },
                        "group_5":
                                  {"inclination":np.deg2rad(97.6),
                                    "planes":4,
                                    "number":172,
                                    "altitude":6371*1e3+560*1e3
                                  }
                }
        bstar=0.00011933
        mean_motion_first_derivative=2.5940898701034544e-14
        mean_motion_second_derivative=0.
        catalog_number_starlink=51955
        if groups=='all':
            for group in range(1,6):
                sats_per_plane=int(starlink_dic[f'group_{group}']['number']/starlink_dic[f'group_{group}']['planes'])
                raans=np.deg2rad(np.arange(0,360,360/starlink_dic[f'group_{group}']['planes']))
                tle_data={}
                k=0
                print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
                for raan in raans:
                    #let's generate the mean anomalies (0-360, for as many sats per plane)
                    mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                    for mean_anomaly in mean_anomalies:
                        k+=1
                        catalog_number_starlink+=k
                        tle_data={}
                        tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                        tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                        tle_data['epoch_year'] = epoch_year
                        tle_data['epoch_days'] = epoch_days
                        tle_data['b_star'] = bstar
                        tle_data['classification'] = 'U'
                        tle_data['international_designator'] = '19029D'
                        tle_data['ephemeris_type'] = 0
                        tle_data['element_number'] = 999
                        tle_data['revolution_number_at_epoch'] = 27384
                        tle_data['satellite_catalog_number']=catalog_number_starlink

                        ##### TLE ELEMENTS #####:
                        tle_data['eccentricity']=0.
                        tle_data['argument_of_perigee']=0.
                        tle_data['inclination']=starlink_dic[f'group_{group}']['inclination']
                        tle_data['mean_motion']=np.sqrt(mu_earth/(starlink_dic[f'group_{group}']['altitude']**3))
                        tle_data['mean_anomaly']=mean_anomaly
                        tle_data['raan']=raan

                        tle=TLE(tle_data)
                        tle.line0=f'0 STARLINK-PHASE 1-GROUP {group}-{k}'
                        tles.append(tle)
            return tles
        else:
            sats_per_plane=int(starlink_dic[f'group_{groups}']['number']/starlink_dic[f'group_{groups}']['planes'])
            raans=np.deg2rad(np.arange(0,360,360/starlink_dic[f'group_{groups}']['planes']))
            tle_data={}
            k=0
            print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
            for raan in raans:
                #let's generate the mean anomalies (0-360, for as many sats per plane)
                mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                for mean_anomaly in mean_anomalies:
                    k+=1
                    catalog_number_starlink+=k
                    tle_data={}
                    tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                    tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                    tle_data['epoch_year'] = epoch_year
                    tle_data['epoch_days'] = epoch_days
                    tle_data['b_star'] = bstar
                    tle_data['classification'] = 'U'
                    tle_data['international_designator'] = '19029D'
                    tle_data['ephemeris_type'] = 0
                    tle_data['element_number'] = 999
                    tle_data['revolution_number_at_epoch'] = 27384
                    tle_data['satellite_catalog_number']=catalog_number_starlink

                    ##### TLE ELEMENTS #####:
                    tle_data['eccentricity']=0.
                    tle_data['argument_of_perigee']=0.
                    tle_data['inclination']=starlink_dic[f'group_{groups}']['inclination']
                    tle_data['mean_motion']=np.sqrt(mu_earth/(starlink_dic[f'group_{groups}']['altitude']**3))
                    tle_data['mean_anomaly']=mean_anomaly
                    tle_data['raan']=raan

                    tle=TLE(tle_data)
                    tle.line0=f'0 STARLINK-PHASE 1-GROUP {groups}-{k}'
                    tles.append(tle)
            return tles
    elif constellation_name=='oneweb':
        bstar=0.0002622869999999722
        mean_motion_first_derivative=2.7598220802876877e-15
        mean_motion_second_derivative=0.
        catalog_number_oneweb=89597

        oneweb_dic={"group_1":
                               {"inclination":np.deg2rad(87.9),
                                "planes":18,
                                "number":720,
                                "altitude":6371*1e3+1200*1e3},
                    "group_2":
                               {"inclination":np.deg2rad(87.9),
                                "planes":36,
                                "number":1764,
                                "altitude":6371*1e3+1200*1e3},
                    "group_3":
                              {"inclination":np.deg2rad(40),
                                "planes":32,
                                "number":23040,
                                "altitude":6371*1e3+1200*1e3
                              },
                    "group_4":
                              {"inclination":np.deg2rad(55),
                                "planes":32,
                                "number":23040,
                                "altitude":6371*1e3+1200*1e3
                              }
                }
        if groups=='all':
            for group in range(1,5):
                sats_per_plane=int(oneweb_dic[f'group_{group}']['number']/oneweb_dic[f'group_{group}']['planes'])
                raans=np.deg2rad(np.arange(0,360,360/oneweb_dic[f'group_{group}']['planes']))
                tle_data={}
                k=0
                print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
                for raan in raans:
                    #let's generate the mean anomalies (0-360, for as many sats per plane)
                    mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                    for mean_anomaly in mean_anomalies:
                        k+=1
                        catalog_number_oneweb+=k
                        tle_data={}
                        tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                        tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                        tle_data['epoch_year'] = epoch_year
                        tle_data['epoch_days'] = epoch_days
                        tle_data['b_star'] = bstar
                        tle_data['classification'] = 'U'
                        tle_data['international_designator'] = '19029D'
                        tle_data['ephemeris_type'] = 0
                        tle_data['element_number'] = 999
                        tle_data['revolution_number_at_epoch'] = 27384
                        tle_data['satellite_catalog_number']=catalog_number_oneweb

                        ##### TLE ELEMENTS #####:
                        tle_data['eccentricity']=0.
                        tle_data['argument_of_perigee']=0.
                        tle_data['inclination']=oneweb_dic[f'group_{group}']['inclination']
                        tle_data['mean_motion']=np.sqrt(mu_earth/(oneweb_dic[f'group_{group}']['altitude']**3))
                        tle_data['mean_anomaly']=mean_anomaly
                        tle_data['raan']=raan

                        tle=TLE(tle_data)
                        tle.line0=f'0 ONEWEB-GROUP {group}-{k}'
                        tles.append(tle)
            return tles
        else:
            sats_per_plane=int(oneweb_dic[f'group_{groups}']['number']/oneweb_dic[f'group_{groups}']['planes'])
            raans=np.deg2rad(np.arange(0,360,360/oneweb_dic[f'group_{groups}']['planes']))
            tle_data={}
            k=0
            print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
            for raan in raans:
                #let's generate the mean anomalies (0-360, for as many sats per plane)
                mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                for mean_anomaly in mean_anomalies:
                    k+=1
                    catalog_number_oneweb+=k
                    tle_data={}
                    tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                    tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                    tle_data['epoch_year'] = epoch_year
                    tle_data['epoch_days'] = epoch_days
                    tle_data['b_star'] = bstar
                    tle_data['classification'] = 'U'
                    tle_data['international_designator'] = '19029D'
                    tle_data['ephemeris_type'] = 0
                    tle_data['element_number'] = 999
                    tle_data['revolution_number_at_epoch'] = 27384
                    tle_data['satellite_catalog_number']=catalog_number_oneweb

                    ##### TLE ELEMENTS #####:
                    tle_data['eccentricity']=0.
                    tle_data['argument_of_perigee']=0.
                    tle_data['inclination']=oneweb_dic[f'group_{groups}']['inclination']
                    tle_data['mean_motion']=np.sqrt(mu_earth/(oneweb_dic[f'group_{groups}']['altitude']**3))
                    tle_data['mean_anomaly']=mean_anomaly
                    tle_data['raan']=raan

                    tle=TLE(tle_data)
                    tle.line0=f'0 ONEWEB-GROUP {groups}-{k}'
                    tles.append(tle)
            return tles

    elif constellation_name=='kuiper':
        kuiper_dic={"group_1":
                            {"inclination":np.deg2rad(33),
                                "planes":28,
                                "number":784,
                                "altitude":6371*1e3+590},
                    "group_2":
                            {"inclination":np.deg2rad(33),
                                "planes":42,
                                "number":1512,
                                "altitude":6371*1e3+610},
                    "group_3":
                            {"inclination":np.deg2rad(33),
                                "planes":34,
                                "number":1156,
                                "altitude":6371*1e3+630}
                }

        #Not knowing precisely these parameters for the constellation, we assume the following (same as Starlink):
        bstar=0.00011933
        mean_motion_first_derivative=2.5940898701034544e-14
        mean_motion_second_derivative=0.
        catalog_number_kuiper=89597
        if groups=='all':
            for group in range(1,4):
                sats_per_plane=int(kuiper_dic[f'group_{group}']['number']/kuiper_dic[f'group_{group}']['planes'])
                raans=np.deg2rad(np.arange(0,360,360/kuiper_dic[f'group_{group}']['planes']))
                tle_data={}
                k=0
                print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
                for raan in raans:
                    #let's generate the mean anomalies (0-360, for as many sats per plane)
                    mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                    for mean_anomaly in mean_anomalies:
                        k+=1
                        catalog_number_kuiper+=k
                        tle_data={}
                        tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                        tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                        tle_data['epoch_year'] = epoch_year
                        tle_data['epoch_days'] = epoch_days
                        tle_data['b_star'] = bstar
                        tle_data['classification'] = 'U'
                        tle_data['international_designator'] = '19029D'
                        tle_data['ephemeris_type'] = 0
                        tle_data['element_number'] = 999
                        tle_data['revolution_number_at_epoch'] = 27384
                        tle_data['satellite_catalog_number']=catalog_number_kuiper

                        ##### TLE ELEMENTS #####:
                        tle_data['eccentricity']=0.
                        tle_data['argument_of_perigee']=0.
                        tle_data['inclination']=kuiper_dic[f'group_{group}']['inclination']
                        tle_data['mean_motion']=np.sqrt(mu_earth/(kuiper_dic[f'group_{group}']['altitude']**3))
                        tle_data['mean_anomaly']=mean_anomaly
                        tle_data['raan']=raan

                        tle=TLE(tle_data)
                        tle.line0=f'0 Kuiper-GROUP {group}-{k}'
                        tles.append(tle)
            return tles
        else:
            sats_per_plane=int(kuiper_dic[f'group_{groups}']['number']/kuiper_dic[f'group_{groups}']['planes'])
            raans=np.deg2rad(np.arange(0,360,360/kuiper_dic[f'group_{groups}']['planes']))
            tle_data={}
            k=0
            print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
            for raan in raans:
                #let's generate the mean anomalies (0-360, for as many sats per plane)
                mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                for mean_anomaly in mean_anomalies:
                    k+=1
                    catalog_number_kuiper+=k
                    tle_data={}
                    tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                    tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                    tle_data['epoch_year'] = epoch_year
                    tle_data['epoch_days'] = epoch_days
                    tle_data['b_star'] = bstar
                    tle_data['classification'] = 'U'
                    tle_data['international_designator'] = '19029D'
                    tle_data['ephemeris_type'] = 0
                    tle_data['element_number'] = 999
                    tle_data['revolution_number_at_epoch'] = 27384
                    tle_data['satellite_catalog_number']=catalog_number_kuiper

                    ##### TLE ELEMENTS #####:
                    tle_data['eccentricity']=0.
                    tle_data['argument_of_perigee']=0.
                    tle_data['inclination']=kuiper_dic[f'group_{groups}']['inclination']
                    tle_data['mean_motion']=np.sqrt(mu_earth/(kuiper_dic[f'group_{groups}']['altitude']**3))
                    tle_data['mean_anomaly']=mean_anomaly
                    tle_data['raan']=raan

                    tle=TLE(tle_data)
                    tle.line0=f'0 Kuiper-GROUP {groups}-{k}'
                    tles.append(tle)
            return tles

    elif constellation_name=='guowang':
        guowang_dic={"group_1":
                            {"inclination":np.deg2rad(85),
                                "planes":16,
                                "number":480,
                                "altitude":6371*1e3+590},
                    "group_2":
                            {"inclination":np.deg2rad(50),
                                "planes":40,
                                "number":2000,
                                "altitude":6371*1e3+600},
                    "group_3":
                            {"inclination":np.deg2rad(55),
                                "planes":60,
                                "number":3600,
                                "altitude":6371*1e3+508},
                    "group_4":
                            {"inclination":np.deg2rad(30),
                                "planes":48,
                                "number":1728,
                                "altitude":6371*1e3+1145},
                    "group_5":
                            {"inclination":np.deg2rad(40),
                                "planes":48,
                                "number":1728,
                                "altitude":6371*1e3+1145},
                    "group_6":
                            {"inclination":np.deg2rad(50),
                                "planes":48,
                                "number":1728,
                                "altitude":6371*1e3+1145},
                    "group_7":
                            {"inclination":np.deg2rad(60),
                                "planes":48,
                                "number":1728,
                                "altitude":6371*1e3+1145}
                }

        #Not knowing precisely these parameters for the constellation, we assume the following (same as Starlink):
        bstar=0.00011933
        mean_motion_first_derivative=2.5940898701034544e-14
        mean_motion_second_derivative=0.
        catalog_number_guowang=89597
        if groups=='all':
            for group in range(1,8):
                sats_per_plane=int(guowang_dic[f'group_{group}']['number']/guowang_dic[f'group_{group}']['planes'])
                raans=np.deg2rad(np.arange(0,360,360/guowang_dic[f'group_{group}']['planes']))
                tle_data={}
                k=0
                print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
                for raan in raans:
                    #let's generate the mean anomalies (0-360, for as many sats per plane)
                    mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                    for mean_anomaly in mean_anomalies:
                        k+=1
                        catalog_number_guowang+=k
                        tle_data={}
                        tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                        tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                        tle_data['epoch_year'] = epoch_year
                        tle_data['epoch_days'] = epoch_days
                        tle_data['b_star'] = bstar
                        tle_data['classification'] = 'U'
                        tle_data['international_designator'] = '19029D'
                        tle_data['ephemeris_type'] = 0
                        tle_data['element_number'] = 999
                        tle_data['revolution_number_at_epoch'] = 27384
                        tle_data['satellite_catalog_number']=catalog_number_guowang

                        ##### TLE ELEMENTS #####:
                        tle_data['eccentricity']=0.
                        tle_data['argument_of_perigee']=0.
                        tle_data['inclination']=guowang_dic[f'group_{group}']['inclination']
                        tle_data['mean_motion']=np.sqrt(mu_earth/(guowang_dic[f'group_{group}']['altitude']**3))
                        tle_data['mean_anomaly']=mean_anomaly
                        tle_data['raan']=raan

                        tle=TLE(tle_data)
                        tle.line0=f'0 Guo Wang-GROUP {group}-{k}'
                        tles.append(tle)
            return tles
        else:
            sats_per_plane=int(guowang_dic[f'group_{groups}']['number']/guowang_dic[f'group_{groups}']['planes'])
            raans=np.deg2rad(np.arange(0,360,360/guowang_dic[f'group_{groups}']['planes']))
            tle_data={}
            k=0
            print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
            for raan in raans:
                #let's generate the mean anomalies (0-360, for as many sats per plane)
                mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                for mean_anomaly in mean_anomalies:
                    k+=1
                    catalog_number_guowang+=k
                    tle_data={}
                    tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                    tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                    tle_data['epoch_year'] = epoch_year
                    tle_data['epoch_days'] = epoch_days
                    tle_data['b_star'] = bstar
                    tle_data['classification'] = 'U'
                    tle_data['international_designator'] = '19029D'
                    tle_data['ephemeris_type'] = 0
                    tle_data['element_number'] = 999
                    tle_data['revolution_number_at_epoch'] = 27384
                    tle_data['satellite_catalog_number']=catalog_number_guowang

                    ##### TLE ELEMENTS #####:
                    tle_data['eccentricity']=0.
                    tle_data['argument_of_perigee']=0.
                    tle_data['inclination']=guowang_dic[f'group_{groups}']['inclination']
                    tle_data['mean_motion']=np.sqrt(mu_earth/(guowang_dic[f'group_{groups}']['altitude']**3))
                    tle_data['mean_anomaly']=mean_anomaly
                    tle_data['raan']=raan

                    tle=TLE(tle_data)
                    tle.line0=f'0 Guo Wang-GROUP {groups}-{k}'
                    tles.append(tle)
            return tles

    elif constellation_name=='satrevolution':
        satrev_dic={"group_1":
                    {"inclination":np.deg2rad(97.5),
                        "planes":1,
                        "number":512,
                        "altitude":6371*1e3+470},
            "group_2":
                    {"inclination":np.deg2rad(60),
                        "planes":1,
                        "number":512,
                        "altitude":6371*1e3+520}
        }
        #Not knowing precisely these parameters for the constellation, we assume the following (same as Starlink):
        bstar=0.00011933
        mean_motion_first_derivative=2.5940898701034544e-14
        mean_motion_second_derivative=0.
        catalog_number_satrev=89597
        if groups=='all':
            for group in range(1,3):
                sats_per_plane=int(satrev_dic[f'group_{group}']['number']/satrev_dic[f'group_{group}']['planes'])
                raans=np.deg2rad(np.arange(0,360,360/satrev_dic[f'group_{group}']['planes']))
                tle_data={}
                k=0
                print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
                for raan in raans:
                    #let's generate the mean anomalies (0-360, for as many sats per plane)
                    mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                    for mean_anomaly in mean_anomalies:
                        k+=1
                        catalog_number_satrev+=k
                        tle_data={}
                        tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                        tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                        tle_data['epoch_year'] = epoch_year
                        tle_data['epoch_days'] = epoch_days
                        tle_data['b_star'] = bstar
                        tle_data['classification'] = 'U'
                        tle_data['international_designator'] = '19029D'
                        tle_data['ephemeris_type'] = 0
                        tle_data['element_number'] = 999
                        tle_data['revolution_number_at_epoch'] = 27384
                        tle_data['satellite_catalog_number']=catalog_number_satrev

                        ##### TLE ELEMENTS #####:
                        tle_data['eccentricity']=0.
                        tle_data['argument_of_perigee']=0.
                        tle_data['inclination']=satrev_dic[f'group_{group}']['inclination']
                        tle_data['mean_motion']=np.sqrt(mu_earth/(satrev_dic[f'group_{group}']['altitude']**3))
                        tle_data['mean_anomaly']=mean_anomaly
                        tle_data['raan']=raan

                        tle=TLE(tle_data)
                        tle.line0=f'0 SatRev-GROUP {group}-{k}'
                        tles.append(tle)
            return tles
        else:
            sats_per_plane=int(satrev_dic[f'group_{groups}']['number']/satrev_dic[f'group_{groups}']['planes'])
            raans=np.deg2rad(np.arange(0,360,360/satrev_dic[f'group_{groups}']['planes']))
            tle_data={}
            k=0
            print(f"number of sats being created: {len(raans)}x{len(np.arange(0,360,360/sats_per_plane))}={len(raans)*len(np.arange(0,360,360/sats_per_plane))}")
            for raan in raans:
                #let's generate the mean anomalies (0-360, for as many sats per plane)
                mean_anomalies=np.deg2rad(np.arange(0,360,360/sats_per_plane))
                for mean_anomaly in mean_anomalies:
                    k+=1
                    catalog_number_satrev+=k
                    tle_data={}
                    tle_data['mean_motion_first_derivative'] = mean_motion_first_derivative
                    tle_data['mean_motion_second_derivative'] = mean_motion_second_derivative
                    tle_data['epoch_year'] = epoch_year
                    tle_data['epoch_days'] = epoch_days
                    tle_data['b_star'] = bstar
                    tle_data['classification'] = 'U'
                    tle_data['international_designator'] = '19029D'
                    tle_data['ephemeris_type'] = 0
                    tle_data['element_number'] = 999
                    tle_data['revolution_number_at_epoch'] = 27384
                    tle_data['satellite_catalog_number']=catalog_number_satrev

                    ##### TLE ELEMENTS #####:
                    tle_data['eccentricity']=0.
                    tle_data['argument_of_perigee']=0.
                    tle_data['inclination']=satrev_dic[f'group_{groups}']['inclination']
                    tle_data['mean_motion']=np.sqrt(mu_earth/(satrev_dic[f'group_{groups}']['altitude']**3))
                    tle_data['mean_anomaly']=mean_anomaly
                    tle_data['raan']=raan

                    tle=TLE(tle_data)
                    tle.line0=f'0 SatRev-GROUP {groups}-{k}'
                    tles.append(tle)
            return tles

def create_path(path, directory=False):
    """
    This function creates a path if it does not exist.

    Args:
        path (`str`): path to be created
        directory (`bool`): if True, the path is a directory, otherwise it is a file
    """
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print('{} does not exist, creating'.format(dir))
        try:
            os.makedirs(dir)
        except Exception as e:
            print(e)
            print('Could not create path, potentially created by another process in the meantime: {}'.format(path))


def create_priors_from_tles(tles, mixture_components = {'mean_motion': 5, 'eccentricity': 5, 'inclination': 13, 'b_star': 4}):
    """
    This function takes a list of TLEs and a dictionary of mixture_components numbers,
    and returns a dictionary of prior distributions for the TLE elements,
    by fitting probability density functions to data using the specified number of mixture components for each element.

    Args:
        `list`: list of `dsgp4.tle.TLE` objects
        `dict`: dictionary of mixture component numbers (`mean_motion`, `eccentricity`,
                                      `inclination` and `b_star` can be selected).

    Returns:
        `dict`: dictionary of prior distributions.
    """
    from pyprob.distributions import Mixture, TruncatedNormal, Uniform
    #I extract the tle elements from the tles:
    tle_els = tle_elements(tles)

    mean_motion = tle_els[0]
    eccentricity = tle_els[1]
    inclination = tle_els[2]
    agument_of_perigee = tle_els[3]
    raan = tle_els[4]
    b_star = tle_els[5]
    mean_anomaly = tle_els[6]
    mean_motion_first_derivative = tle_els[7]
    #mean_motion_second_derivative = tle_els[8]

    priors_dict = {}
    m = fit_mixture(np.array(mean_motion)*10000, n_components = mixture_components['mean_motion'], covariance_type = 'diag')
    dists = []
    for i in range(len(m.means_)):
        dists.append(TruncatedNormal(mean_non_truncated = m.means_[i][0]/10000, stddev_non_truncated = np.sqrt(m.covariances_[i][0])/10000, low = min(mean_motion), high = max(mean_motion)))
    priors_dict['mean_motion_prior'] = Mixture(distributions = dists, probs = list(m.weights_))

    m = fit_mixture(values = np.array(eccentricity), n_components = mixture_components['eccentricity'], covariance_type = 'diag')
    dists = []
    for i in range(len(m.means_)):
        dists.append(TruncatedNormal(mean_non_truncated = m.means_[i][0], stddev_non_truncated = np.sqrt(m.covariances_[i][0]), low = 0., high = max(eccentricity) ))
    priors_dict['eccentricity_prior'] = Mixture(distributions = dists, probs = list(m.weights_))

    m = fit_mixture(values = np.array(inclination), n_components = mixture_components['inclination'], covariance_type = 'diag')
    dists = []
    for i in range(len(m.means_)):
        dists.append(TruncatedNormal(mean_non_truncated = m.means_[i][0], stddev_non_truncated = np.sqrt(m.covariances_[i][0]), low = 0., high = np.pi ))
    priors_dict['inclination_prior'] = Mixture(distributions = dists, probs = list(m.weights_))

    m = fit_mixture(values = np.array(b_star)*10000, n_components = mixture_components['b_star'], covariance_type = 'diag')
    dists = []
    for i in range(len(m.means_)):
        dists.append(TruncatedNormal(mean_non_truncated = m.means_[i][0]/10000, stddev_non_truncated = np.sqrt(m.covariances_[i][0]), low = min(b_star), high = max(b_star) ))
    priors_dict['b_star_prior'] = Mixture(distributions = dists, probs = list(m.weights_))
#    if plot==True:
#        analysis.plot_mix(priors_dict)
    priors_dict['mean_anomaly_prior'] = Uniform(low=0.0, high=2*np.pi)
    priors_dict['argument_of_perigee_prior'] = Uniform(low=0.0, high=2*np.pi)
    priors_dict['raan_prior'] = Uniform(low=0.0, high=2*np.pi)
    priors_dict['mean_motion_first_derivative_prior'] = TruncatedNormal(mean_non_truncated = np.mean(mean_motion_first_derivative), stddev_non_truncated = np.std(mean_motion_first_derivative), low = min(mean_motion_first_derivative), high = max(mean_motion_first_derivative))
    return priors_dict


def tle_elements(tles):
    """
    This function takes a list of TLEs as input and extracts their elements as lists.

    Args:
        - tles (`list`): list of `dsgp4.tle.TLE` objects
    Returns:
        - mean_motion, eccentricity, inclination, argument_of_perigee, raan, b_star, mean_anomaly, mean_motion_first_derivative, mean_motion_second_derivative
    Example::
        import matplotlib.pyplot as plt
        import kessler
        sats = dsgp4.tle.load(file_name = 'path_to_tle.txt')
        n, e, i, omega, RAAN, B_star, M, n_dot, n_ddot = dsgp4.tle.tle_elements(sats)#tles is a list of TLEs dictionary
        plt.hist(n)
    """
    mean_motion, eccentricity, inclination, argument_of_perigee, raan, b_star, mean_anomaly, mean_motion_first_derivative, mean_motion_second_derivative = [], [], [], [], [], [], [], [], []
    for tle in tles:
        mean_motion.append(tle.mean_motion)
        eccentricity.append(tle.eccentricity)
        inclination.append(tle.inclination)
        argument_of_perigee.append(tle.argument_of_perigee)
        raan.append(tle.raan)
        b_star.append(tle.b_star)
        mean_anomaly.append(tle.mean_anomaly)
        mean_motion_first_derivative.append(tle.mean_motion_first_derivative)
        mean_motion_second_derivative.append(tle.mean_motion_second_derivative)
    return mean_motion, eccentricity, inclination, argument_of_perigee, raan, b_star, mean_anomaly, mean_motion_first_derivative, mean_motion_second_derivative


def add_megaconstellation_from_file(tles, megaconstellation_file_name):
    """
    This function takes a list of TLEs and a megaconstellation file name, and returns
    a list of the original TLEs plus the TLEs of the added megaconstellation.

    Args:
        tles (`list`): list of `dsgp4.tle.TLE` objects
        megaconstellation_file_name (`str`): megaconstellation file name

    Returns:
        `list`: list of `dsgp4.tle.TLE` objects
    """
    tles_megaconstellation=dsgp4.util.load(file_name=megaconstellation_file_name)
    return tles+tles_megaconstellation
