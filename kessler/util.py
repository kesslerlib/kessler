import numpy as np
import torch
import math
import os
import pykep
import skyfield
import skyfield.sgp4lib
import datetime
import functools


# This function is from python-sgp4 released under MIT License, (c) 2012–2016 Brandon Rhodes
def compute_checksum(line):
    return sum((int(c) if c.isdigit() else c == '-') for c in line[0:68]) % 10


# Parts of this function is based on python-sgp4 released under MIT License, (c) 2012–2016 Brandon Rhodes
def tle(satnum, classification, international_designator, epoch_year, epoch_days, mean_motion_first_derivative, mean_motion_second_derivative, b_star, ephemeris_type, element_number, inclination, raan, eccentricity, argument_of_perigee, mean_anomaly, mean_motion, revolution_number_at_epoch):
    line1 = ['1 ']
    line1.append(str(satnum).zfill(5)[:5])
    line1.append(str(classification)[0] + ' ')
    line1.append(str(international_designator).ljust(8, ' ')[:8] + ' ')
    line1.append(str(epoch_year)[-2:].zfill(2) + '{:012.8f}'.format(epoch_days) + ' ')
    line1.append('{0: 8.8f}'.format(mean_motion_first_derivative * (1.86624e9 / math.pi)).replace('0', '', 1) + ' ')
    line1.append('{0: 4.4e}'.format((mean_motion_second_derivative * (5.3747712e13 / math.pi)) * 10).replace(".", '').replace('e+00', '-0').replace('e-0', '-').replace('e+0', '+') + ' ')
    line1.append('{0: 4.4e}'.format(b_star * 10).replace('.', '').replace('e+00', '+0').replace('e-0', '-') + ' ')
    line1.append('{} '.format(ephemeris_type) + str(element_number).rjust(4, ' '))
    line1 = ''.join(line1)
    line1 += str(compute_checksum(line1))

    line2 = ['2 ']
    line2.append(str(satnum).zfill(5)[:5] + ' ')
    line2.append('{0:8.4f}'.format(inclination * (180 / math.pi)).rjust(8, ' ') + ' ')
    line2.append('{0:8.4f}'.format(raan * (180 / math.pi)).rjust(8, ' ') + ' ')
    line2.append(str(int(eccentricity * 1e7)).rjust(7, '0')[:7] + ' ')
    line2.append('{0:8.4f}'.format(argument_of_perigee * (180 / math.pi)).rjust(8, ' ') + ' ')
    line2.append('{0:8.4f}'.format(mean_anomaly * (180 / math.pi)).rjust(8, ' ') + ' ')
    line2.append('{0:11.8f}'.format(mean_motion * 43200.0 / math.pi).rjust(8, ' '))
    line2.append(str(revolution_number_at_epoch).rjust(5))
    line2 = ''.join(line2)
    line2 += str(compute_checksum(line2))

    if len(line1) != 69:
        raise RuntimeError('TLE line 1 has unexpected length ({})'.format(len(line1)))
    if len(line2) != 69:
        raise RuntimeError('TLE line 2 has unexpected length ({})'.format(len(line2)))

    return line1, line2


def from_cartesian_to_tle_elements(state):
    r, v = state[0], state[1]
    kepl_el = pykep.ic2par(r, v, pykep.MU_EARTH)
    # these are returned as (a,e,i,W,w,E) --> [m], [-], [rad], [rad], [rad], [rad]
    mean_motion         = np.sqrt(pykep.MU_EARTH/((kepl_el[0])**(3.0)))
    eccentricity        = kepl_el[1]
    inclination         = kepl_el[2]
    argument_of_perigee = kepl_el[4]
    raan                = kepl_el[3]
    mean_anomaly        = kepl_el[5] - kepl_el[1]*np.sin(kepl_el[5])+np.pi
    return mean_motion, eccentricity, inclination, argument_of_perigee, raan, mean_anomaly


def rotation_matrix(state):
    r, v = state[0], state[1]
    u = r / np.linalg.norm(r)
    w = np.cross(r, v)
    w = w / np.linalg.norm(w)
    v = np.cross(w, u)
    return np.vstack((u, v, w))


def from_cartesian_to_rtn(state, cartesian_to_rtn_rotation_matrix=None):
    # Use the supplied rotation matrix if available, otherwise compute it
    if cartesian_to_rtn_rotation_matrix is None:
        cartesian_to_rtn_rotation_matrix = rotation_matrix(state)
    r, v = state[0], state[1]
    r_rtn = np.dot(cartesian_to_rtn_rotation_matrix, r)
    v_rtn = np.dot(cartesian_to_rtn_rotation_matrix, v)
    return np.stack([r_rtn, v_rtn]), cartesian_to_rtn_rotation_matrix


def from_rtn_to_cartesian(state_rtn, rtn_to_cartesian_rotation_matrix):
    r_rtn, v_rtn = state_rtn[0], state_rtn[1]
    state_xyz = np.stack([np.matmul(rtn_to_cartesian_rotation_matrix, r_rtn), np.matmul(rtn_to_cartesian_rotation_matrix, v_rtn)])
    return state_xyz


def from_TEME_to_ITRF(state, time):
    r, v = state[0], state[1]
    # time must be in J2000
    # velocity in the converter is in m/days, so we multiply by 86400 before conversion and divide later
    # print(f'pos: {r}, vel: {v}')
    r_new, v_new = skyfield.sgp4lib.TEME_to_ITRF(time, r, v*86400.)
    # print(f'pos: {r_new}, vel: {v_new/86400.}')
    v_new = v_new / 86400.
    state = np.stack([r_new, v_new])
    return state


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


def from_datetime_to_cdm_datetime_str(datetime):
    return datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')


def from_jd_to_datetime(jd_date):
    e = pykep.epoch(jd_date, 'jd')
    return datetime.datetime.strptime(str(e), '%Y-%b-%d %H:%M:%S.%f')


def from_mjd_to_datetime(mjd_date):
    e = pykep.epoch(mjd_date, 'mjd')
    return datetime.datetime.strptime(str(e), '%Y-%b-%d %H:%M:%S.%f')


def from_jd_to_cdm_datetime_str(jd_date):
    d = from_jd_to_datetime(jd_date)
    return from_datetime_to_cdm_datetime_str(d)


def from_mjd_to_epoch_days_after_1_jan(mjd_date):
    d = from_mjd_to_datetime(mjd_date)
    dd = d - datetime.datetime(d.year, 1, 1)
    days = dd.days
    days_fraction = (dd.seconds + dd.microseconds/1e6) / (60*60*24)
    return days + days_fraction


@functools.lru_cache(maxsize=None)
def from_date_str_to_days(date, date0='2020-05-22T21:41:31.975'):
    date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f')
    date0 = datetime.datetime.strptime(date0, '%Y-%m-%dT%H:%M:%S.%f')
    dd = date-date0
    days = dd.days
    days_fraction = (dd.seconds + dd.microseconds/1e6) / (60*60*24)
    return days + days_fraction


def add_days_to_date_str(date0, days):
    date0 = datetime.datetime.strptime(date0, '%Y-%m-%dT%H:%M:%S.%f')
    date = date0 + datetime.timedelta(days=days)
    return from_datetime_to_cdm_datetime_str(date)


pykep_satellite = None


def lpop_init(tle):
    global pykep_satellite
    pykep_satellite = pykep.planet.tle(tle[0], tle[1])


def lpop_single(target_mjd):
    return pykep_satellite.eph(pykep.epoch(target_mjd, 'mjd'))


def lpop_sequence(target_mjds):
    return np.array(list(map(lpop_single, target_mjds)))


def lpop_sequence_upsample(target_mjds, upsample_factor=1):
    if upsample_factor == 1:
        return lpop_sequence(target_mjds)
    else:
        take_every = max(upsample_factor, 1)
        target_mjds_subsample = [target_mjds[i] for i in range(0, len(target_mjds), take_every)]
        ret = lpop_sequence(target_mjds_subsample)
        ret = torch.from_numpy(ret).view(ret.shape[0], -1)
        ret = upsample(ret, len(target_mjds))
        ret = ret.view(ret.shape[0], 2, 3).cpu().numpy()
        return ret


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
