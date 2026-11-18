import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from refine_clock_maps import build_map,map_times,predict_lag

class LocalClockTests(unittest.TestCase):
    def model(self):
        return build_map(pd.DataFrame({'center_hours':[1.,2.,3.],
            'lag_other_minus_reference_s':[.2,.3,.5],
            'timing_signal_supported':[True,True,True]}))

    def test_known_segment_inverse_and_no_extrapolation(self):
        m=self.model()
        ref=np.array([3600.,5400.,7200.,9000.,10800.])
        lag=predict_lag(ref,m)
        native=ref+lag
        original=native.copy()
        mapped,inside=map_times(native,m)
        np.testing.assert_allclose(mapped,ref,atol=1e-9)
        self.assertTrue(inside.all())
        np.testing.assert_array_equal(native,original)
        outside,flags=map_times(np.array([0.,20000.]),m)
        self.assertFalse(flags.any())
        self.assertTrue(np.isnan(outside).all())

    def test_nonmonotonic_clock_knots_are_rejected(self):
        data=pd.DataFrame({'center_hours':[1.,2.,3.],
            'lag_other_minus_reference_s':[.2,-4000.,.3],
            'timing_signal_supported':[True]*3})
        with self.assertRaises(ValueError):build_map(data)

    def test_unverified_anchor_and_outside_prediction_are_rejected(self):
        data=pd.DataFrame({'center_hours':[1.,2.,3.],
            'lag_other_minus_reference_s':[.2,.3,.5],
            'timing_signal_supported':[True,False,True]})
        with self.assertRaises(ValueError):build_map(data)
        with self.assertRaises(ValueError):predict_lag(np.array([0.]),self.model())

if __name__=='__main__':unittest.main()
