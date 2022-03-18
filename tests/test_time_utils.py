# This test is written by Andrew Ng, 19/03/22. It makes use of several CDMs provided by the NASA CARA
# analysis repo at https://github.com/nasa/CARA_Analysis_Tools/tree/master/two-dimension_Pc/UnitTest/InputFiles.
# This test verifies the proper functioning of getCcsdsTimeFormat and DOY2Date. The selected CDMs ran successfully with 
# the MATLAB code provided by the CARA_Analysis_Tools repo, so they need to work here as well. 
import unittest
from kessler import time_utils
import datetime
class Test_time_utils(unittest.TestCase):
    def test_getCcsdsTimeFormat(self):
        test_case1 = "2000-01-01T00:00:00.000" #From AlfanoTestCase11.cdm
        test_case2 = "2018-229T13:56:33.000" # From DensityDecorrelationTestCaseCDM.txt
        test_case1_correct = "yyyy-mm-ddTHH:MM:SS.FFF"
        test_case2_correct = "yyyy-DDDTHH:MM:SS.FFF"

        self.assertEqual(time_utils.getCcsdsTimeFormat(test_case1), test_case1_correct)
        self.assertEqual(time_utils.getCcsdsTimeFormat(test_case2), test_case2_correct) 
    def test_DOY2Date(self):
        example1 = "2010-202T12:25:19.000" # From SingleCovTestCase1-4.cdm
        example2 = "2018-229T13:56:33.000" # From DensityDecorrelationTestCaseCDM.txt
        DOY_1 = example1[5:5+3] 
        Year_1= example1[0:4]
        DOY_2 = example2[5:5+3]
        Year_2= example2[0:4]
        test_case1_correct = datetime.datetime(2010, 7, 21, 0, 0), [2010, 7, 21, 0, 0]
        test_case2_correct = datetime.datetime(2018, 8, 17, 0, 0), [2018, 8, 17, 0, 0]
        self.assertEqual(time_utils.DOY2Date(DOY_1, Year_1), test_case1_correct)
        self.assertEqual(time_utils.DOY2Date(DOY_2, Year_2), test_case2_correct) 

if __name__ == '__main__':
    unittest.main()