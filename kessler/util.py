import numpy as np
import torch
import math
import pykep


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


def upsample(s, target_resolution):
    s = s.transpose(0, 1)
    s = torch.nn.functional.interpolate(s.unsqueeze(0), size=(target_resolution), mode='linear', align_corners=True)
    s = s.squeeze(0).transpose(0, 1)
    return s


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
