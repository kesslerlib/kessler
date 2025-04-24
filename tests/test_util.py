# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# Trillium Technologies
# University of Oxford
# Giacomo Acciarini (giacomo.acciarini@gmail.com)
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.

import unittest
import numpy as np
import datetime
import torch
import dsgp4
import kessler
import kessler.util

from pyro.distributions import Categorical, MixtureSameFamily
class UtilTestCase(unittest.TestCase):
    def test_from_datetime_to_cdm_datetime_str(self):
        date = datetime.datetime(2823, 3, 4, 12, 1, 23, 252 )
        date_str = kessler.util.from_datetime_to_cdm_datetime_str(date)
        date_str_correct = '2823-03-04T12:01:23.000252'
        self.assertEqual(date_str_correct, date_str)

    def test_from_jd_to_cdm_datetime_str(self):
        jd_date = 2469808.0229167
        date_str = kessler.util.from_jd_to_cdm_datetime_str(jd_date)
        date_str_correct = '2050-01-01T12:33:00.002899'
        self.assertEqual(date_str_correct, date_str)

    def test_from_TEME_to_ITRF(self):
        state_TEME = np.array([[5094.18016210*1e3, 6127.64465950*1e3, 6380.34453270*1e3], [-4.746131487*1e3, 0.785818041*1e3, 5.531931288*1e3]])

        jd_date = 2433282.4235
        state_ITRF = kessler.util.from_TEME_to_ITRF(state_TEME, jd_date)
        state_ITRF_correct = np.array([np.array([ 7377976.66089733,-3010674.50719708, 6380344.5327]), np.array([-900.58420598,4224.28457883,5531.931288])])
        print('state_ITRF', state_ITRF)
        print('state_ITRF_correct', state_ITRF_correct)
        print('state_TEME', state_TEME)
        self.assertTrue(np.allclose(state_ITRF_correct, state_ITRF))
        self.assertAlmostEqual(np.linalg.norm(state_TEME[0]), np.linalg.norm(state_ITRF[0]), places=1)

    def test_from_cartesian_to_rtn(self):
        state_xyz = np.array([[1.0, 2.0, 3.4], [4.5, 6.2, 7.4]])
        state_rtn, cartesian_to_rtn_rotation_matrix = kessler.util.from_cartesian_to_rtn(state_xyz)
        rtn_to_cartesian_rotation_matrix = cartesian_to_rtn_rotation_matrix.T
        r_xyz, v_xyz = np.matmul(rtn_to_cartesian_rotation_matrix, state_rtn[0]), np.matmul(rtn_to_cartesian_rotation_matrix, state_rtn[1])

        self.assertTrue(np.allclose(state_xyz[0], r_xyz))
        self.assertTrue(np.allclose(state_xyz[1], v_xyz))
        self.assertAlmostEqual(np.linalg.norm(state_xyz[0]), np.linalg.norm(state_rtn[0]), places=1)
        self.assertAlmostEqual(np.linalg.norm(state_xyz[1]), np.linalg.norm(state_rtn[1]), places=1)

    def test_from_cartesian_to_rtn_2(self):
        state_xyz = np.array([[1.0, 2.0, 3.4], [4.5, 6.2, 7.4]])
        state_rtn, _ = kessler.util.from_cartesian_to_rtn(state_xyz)
        self.assertAlmostEqual(np.linalg.norm(state_xyz[0]), np.linalg.norm(state_rtn[0]), places=1)
        self.assertAlmostEqual(np.linalg.norm(state_xyz[1]), np.linalg.norm(state_rtn[1]), places=1)

    def test_get_ccsds_time_format(self):
        # This test is written by Andrew Ng, 19/03/22. It makes use of example CDMs provided by the NASA CARA
        # analysis repo at https://github.com/nasa/CARA_Analysis_Tools/tree/master/two-dimension_Pc/UnitTest/InputFiles.
        test_case1 = "2000-01-01T00:00:00.000" #From AlfanoTestCase11.cdm
        test_case2 = "2018-229T13:56:33.000" # From DensityDecorrelationTestCaseCDM.txt
        test_case1_correct = "yyyy-mm-ddTHH:MM:SS.FFF"
        test_case2_correct = "yyyy-DDDTHH:MM:SS.FFF"

        self.assertEqual(kessler.util.get_ccsds_time_format(test_case1), test_case1_correct)
        self.assertEqual(kessler.util.get_ccsds_time_format(test_case2), test_case2_correct) 

    def test_doy_2_date(self):
        # This test is written by Andrew Ng, 19/03/22. It makes use of example CDMs provided by the NASA CARA
        # analysis repo at https://github.com/nasa/CARA_Analysis_Tools/tree/master/two-dimension_Pc/UnitTest/InputFiles.
        example1 = "2010-202T12:25:19.000" # From SingleCovTestCase1-4.cdm
        example2 = "2018-229T13:56:33.000" # From DensityDecorrelationTestCaseCDM.txt
        example3 = "2010-365T00:00:00.000" # Check that works at the final day of a non leap year
        example4 = "2010-001T00:00:00.000" # Check that works at the first day of a year
        example5 = "2012-366T00:00:00.000" # Check that works at the final day of a leap year
        
        doy_1 = example1[5:5+3] 
        year_1= example1[0:4]
        doy_2 = example2[5:5+3]
        year_2= example2[0:4]
        doy_3 = example3[5:5+3] 
        year_3= example3[0:4]
        doy_4 = example4[5:5+3]
        year_4= example4[0:4]
        doy_5 = example5[5:5+3] 
        year_5= example5[0:4]

        test_case1_correct = "2010-07-21T12:25:19.00"
        test_case2_correct = "2018-08-17T13:56:33.00"
        test_case3_correct = "2010-12-31T00:00:00.00"
        test_case4_correct = "2010-01-01T00:00:00.00"
        test_case5_correct = "2012-12-31T00:00:00.00"
        
        self.assertEqual(kessler.util.doy_2_date(example1, doy_1, year_1, 5), test_case1_correct)
        self.assertEqual(kessler.util.doy_2_date(example2, doy_2, year_2, 5), test_case2_correct) 
        self.assertEqual(kessler.util.doy_2_date(example3, doy_3, year_3, 5), test_case3_correct)
        self.assertEqual(kessler.util.doy_2_date(example4, doy_4, year_4, 5), test_case4_correct) 
        self.assertEqual(kessler.util.doy_2_date(example5, doy_5, year_5, 5), test_case5_correct) 
    
    def test_from_cartesian_to_keplerian_torch(self):
        lines=[]
        lines.append("0 COSMOS 2251 DEB")
        lines.append("1 34427U 93036RU  22068.94647328  .00008100  00000-0  11455-2 0  9999")
        lines.append("2 34427  74.0145 306.8269 0033346  13.0723 347.1308 14.76870515693886")
        tle=dsgp4.TLE(lines)
        dsgp4.initialize_tle(tle)
        # Extract the state vector from the dSGP4 output
        st=dsgp4.sgp4(tle,torch.tensor(0.0))*1e3  # Convert to meters and meters/second
        #now let's retrieve the poliastro gravitational parameter of the Earth:
        mu = 398600441800000.0000000000000000#Earth.k.to(u.m**3 / u.s**2).value
        # Let's then convert Cartesian -> Keplerian using our function
        a,e,i,Omega,omega,M=kessler.util.from_cartesian_to_keplerian(r_vec=st[0], v_vec=st[1], mu=mu)

        a_poliastro=7023679.5817881366237998
        e_poliastro=0.0041649630912143
        i_poliastro=1.2919744331609118
        Omega_poliastro=5.3551396410293757
        omega_poliastro=0.4409272281996022
        M_poliastro=5.8458093777349722
        #let's now test they are close:
        self.assertAlmostEqual(a.item(), a_poliastro, places=5)
        self.assertAlmostEqual(e.item(), e_poliastro, places=5)
        self.assertAlmostEqual(i.item(), i_poliastro, places=5)
        self.assertAlmostEqual(Omega.item(), Omega_poliastro, places=5)
        self.assertAlmostEqual(omega.item(), omega_poliastro, places=5)
        self.assertAlmostEqual(M.item(), M_poliastro, places=5)
    
    def test_TruncatedNormal(self):
        #we check the truncated normal distribution, we do this for a mixture of them:         
        locs=torch.tensor([6.391167644720491e-05, 0.021593032643530245,0.00089714840255561, 0.004279950440413096, ]) 
        scales=torch.tensor([9.155414891172675e-05, 0.08825398369676822, 0.0006834100202961423,0.003464680595037937]) 
        probs=torch.tensor([0.36699734 ,0.02809785 ,0.39047779,0.21442703 ]) 
        min=-0.73577 
        max=0.68639

        categorical=Categorical(probs=probs)
        batched_truncated_normal = kessler.util.TruncatedNormal(loc=locs, scale=scales, low=min, high=max)
        mix_truncated=MixtureSameFamily(categorical, batched_truncated_normal)
        self.assertAlmostEqual(mix_truncated.log_prob(torch.tensor(0.0001)).item(), 7.382209300994873, places=6)
        self.assertAlmostEqual(mix_truncated.log_prob(torch.tensor(0.001)).item(), 5.485926151275635, places=6)
        self.assertAlmostEqual(mix_truncated.log_prob(torch.tensor(0.01)).item(), 1.863307237625122, places=6)
        self.assertAlmostEqual(mix_truncated.log_prob(torch.tensor(0.1)).item(), -2.458112955093384, places=6)


