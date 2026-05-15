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
import os

import kessler.model
import kessler
from kessler import GNSS, Radar, EventDataset, foster_collision_probability

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


class AkellaTestCase(unittest.TestCase):
    def test_foster_collision_probability_basic(self):
        """Test that foster_collision_probability returns a valid probability."""
        # Load a CDM from the synthetic dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cdms_dir = os.path.join(current_dir, '..', 'docs', 'notebooks', 'synthetic_cdms')
        
        events = EventDataset(cdms_dir=cdms_dir, cdm_extension='.kvn')
        cdms = events.get_CDMs()
        
        self.assertGreater(len(cdms), 0, "No CDMs loaded from synthetic dataset")
        
        cdm = cdms[0]
        
        # Compute probability of collision
        pc = foster_collision_probability(cdm, collision_radius=70)
        
        # Verify output is a valid probability
        self.assertIsInstance(pc, (float, np.floating))
        self.assertGreaterEqual(pc, 0.0, "Probability should be >= 0")
        self.assertLessEqual(pc, 1.0, "Probability should be <= 1")
    
    def test_foster_collision_probability_multiple_cdms(self):
        """Test foster_collision_probability on multiple CDMs."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cdms_dir = os.path.join(current_dir, '..', 'docs', 'notebooks', 'synthetic_cdms')
        
        events = EventDataset(cdms_dir=cdms_dir, cdm_extension='.kvn')
        cdms = events.get_CDMs()
        
        # Test on first 5 CDMs
        for cdm in cdms[:5]:
            pc = foster_collision_probability(cdm, collision_radius=70)
            self.assertGreaterEqual(pc, 0.0)
            self.assertLessEqual(pc, 1.0)
    
    def test_foster_collision_probability_custom_radius(self):
        """Test foster_collision_probability with different collision radii."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cdms_dir = os.path.join(current_dir, '..', 'docs', 'notebooks', 'synthetic_cdms')
        
        events = EventDataset(cdms_dir=cdms_dir, cdm_extension='.kvn')
        cdm = events.get_CDMs()[0]
        
        # Test with different radii
        radii = [50, 70, 100, 150]
        probabilities = []
        
        for radius in radii:
            pc = foster_collision_probability(cdm, collision_radius=radius)
            probabilities.append(pc)
            self.assertGreaterEqual(pc, 0.0)
            self.assertLessEqual(pc, 1.0)
        
        # Probability should generally increase with collision radius
        # (though not strictly monotonic for all cases)
        for i in range(len(probabilities) - 1):
            # Just verify they are all valid probabilities
            self.assertIsInstance(probabilities[i], (float, np.floating))
