# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# Trillium Technologies
# University of Oxford
# Giacomo Acciarini (giacomo.acciarini@gmail.com)
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.

import numpy as np
import unittest
import dsgp4

import kessler.model
from kessler import GNSS, Radar

class UtilTestCase(unittest.TestCase):
    def test_make_chaser_make_target(self):
        t_tle_list=['0 ELECTRON KICK STAGE R/B',
               '1 44227U 19026C   22068.79876951  .00010731  00000-0  41303-3 0  9993',
               '2 44227  40.0221 252.2030 0008096   5.2961 354.7926 15.26135826158481']

        c_tle_list=['0 HARBINGER',
               '1 44229U 19026E   22068.90017356  .00004812  00000-0  20383-3 0  9992',
               '2 44229  40.0180 261.5261 0008532 356.1827   3.8908 15.23652474158314']
        t_tle = dsgp4.tle.TLE(t_tle_list)
        c_tle = dsgp4.tle.TLE(c_tle_list)
        model = kessler.model.Conjunction(t_observing_instruments=[GNSS()], c_observing_instruments=[Radar()], t_tle=t_tle, c_tle=c_tle)
        model_tle_target = model.make_target()
        model_tle_chaser = model.make_chaser()

        model_tle_target.update({"mean_anomaly": float(t_tle.mean_anomaly)})
        model_tle_chaser.update({"mean_anomaly": float(c_tle.mean_anomaly)})

        self.assertEqual(model_tle_target.line1, t_tle_list[1])
        self.assertEqual(model_tle_target.line2, t_tle_list[2])
        self.assertEqual(model_tle_chaser.line1, c_tle_list[1])
        self.assertEqual(model_tle_chaser.line2, c_tle_list[2])
