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

from kessler import GNSS, Radar
 
class UtilTestCase(unittest.TestCase):
    def test_radar(self):
        spacecraft = {}
        spacecraft['state_xyz']= np.array([[1002.3, 202.2, 103.0], [0.1, 0.23, 0.4]])
        instrument_characteristics = {}
        instrument_characteristics['bias_xyz']= np.zeros((2,3))
        instrument_characteristics['covariance_rtn'] = np.array([1e-9, 1.115849341564346, 0.059309835843067, 1e-9, 1e-9, 1e-9])**2
        instrument = Radar(instrument_characteristics)
        state_observed, covariance_rtn = instrument.observe(time=0., spacecraft=spacecraft)
        self.assertTrue(np.allclose(state_observed, spacecraft['state_xyz']))
        self.assertTrue(np.allclose(covariance_rtn, instrument_characteristics['covariance_rtn']))
        instrument_characteristics_2 = {}
        instrument_characteristics_2['bias_xyz']= np.array([[0.2, 30.3, 40.4],[102.3, 404.4, 22.2]])
        instrument_characteristics_2['covariance_rtn'] = np.array([1e-9, 1.115849341564346, 0.059309835843067, 1e-9, 1e-9, 1e-9])**2
        instrument_2 = Radar(instrument_characteristics_2)
        state_observed, _ = instrument_2.observe(time=0., spacecraft=spacecraft)
        self.assertTrue(np.allclose(state_observed, spacecraft['state_xyz']+instrument_characteristics_2['bias_xyz']))

    def test_gnss(self):
        spacecraft = {}
        spacecraft['state_xyz']= np.array([[1002.3, 202.2, 103.0], [0.1, 0.23, 0.4]])
        instrument_characteristics = {}
        instrument_characteristics['bias_xyz']= np.zeros((2,3))
        instrument_characteristics['covariance_rtn'] = np.array([1e-9, 1.115849341564346, 0.059309835843067, 1e-9, 1e-9, 1e-9])**2
        instrument = GNSS(instrument_characteristics)
        state_observed, covariance_rtn = instrument.observe(time=0., spacecraft=spacecraft)
        self.assertTrue(np.allclose(state_observed, spacecraft['state_xyz']))
        self.assertTrue(np.allclose(covariance_rtn, instrument_characteristics['covariance_rtn']))
        instrument_characteristics_2 = {}
        instrument_characteristics_2['bias_xyz']= np.array([[0.2, 30.3, 40.4],[102.3, 404.4, 22.2]])
        instrument_characteristics_2['covariance_rtn'] = np.array([1e-9, 1.115849341564346, 0.059309835843067, 1e-9, 1e-9, 1e-9])**2
        instrument_2 = GNSS(instrument_characteristics_2)
        state_observed, _ = instrument_2.observe(time=0., spacecraft=spacecraft)
        self.assertTrue(np.allclose(state_observed, spacecraft['state_xyz']+instrument_characteristics_2['bias_xyz']))
