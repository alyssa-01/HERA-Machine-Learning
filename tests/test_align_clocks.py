"""Checks for sign, units, validation separation, and unsupported timing signals."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from align_clocks import local_lag, fit_clock, map_to_reference, validate_split


class ClockAlignmentTests(unittest.TestCase):
    def test_inverse_map_sign_units_and_input_preservation(self):
        model={'offset_s':.36,'drift':4e-6,
               'training_center_domain_reference_s':[1000.,5000.]}
        ref=np.array([0.,1000.,3000.,5000.,6000.])
        native=.36+(1+4e-6)*ref
        before=native.copy()
        mapped,inside=map_to_reference(native,model)
        np.testing.assert_allclose(mapped,ref,atol=1e-9)
        np.testing.assert_array_equal(native,before)
        np.testing.assert_array_equal(inside,[False,True,True,True,False])

    def test_windows_must_be_separated(self):
        with self.assertRaises(ValueError):
            validate_split([1000,3000,5000],[1100,4000],600,120)

    def test_validation_rows_do_not_change_coefficients(self):
        train=pd.DataFrame({'center_hours':[1.,2.,3.,4.],
                            'lag_other_minus_reference_s':[.4,.44,.48,.52],
                            'timing_signal_supported':[True]*4})
        val=pd.DataFrame({'center_hours':[1.5,2.5],
                          'lag_other_minus_reference_s':[.42,.46],
                          'timing_signal_supported':[True]*2})
        first,_=fit_clock(train,val)
        val['lag_other_minus_reference_s']=[-100.,100.]
        second,_=fit_clock(train,val)
        self.assertEqual(first,second)

    def test_injected_offset_and_drift(self):
        rng=np.random.default_rng(381)
        duration=6500.
        shared=np.sort(rng.uniform(0,duration,12000))
        ref=np.sort(np.r_[shared,rng.uniform(0,duration,32000)])
        offset,drift=.372,5e-6
        other_real=np.sort(np.r_[shared+rng.normal(0,.00005,len(shared)),
                                rng.uniform(0,duration,34000)])
        other=offset+(1+drift)*other_real
        rows=[]
        for center in [1000.,3000.,5000.]:
            row,_=local_lag(ref,other,center,width_s=240.,search_s=2.,bin_s=.01)
            self.assertTrue(row['timing_signal_supported'])
            self.assertLess(abs(row['lag_other_minus_reference_s']-(offset+drift*center)),.003)
            rows.append(row)
        train=pd.DataFrame(rows)
        holdout,_=local_lag(ref,other,4000.,width_s=240.,search_s=2.,bin_s=.01)
        model,metrics=fit_clock(train,pd.DataFrame([holdout]))
        self.assertLess(abs(model['drift']-drift),1e-6)
        self.assertLess(metrics['affine_validation_mae_ms'],3.)

    def test_independent_event_stream_has_no_supported_peak(self):
        rng=np.random.default_rng(91)
        ref=np.sort(rng.uniform(0,10000,60000))
        other=np.sort(rng.uniform(0,10000,60000))
        row,_=local_lag(ref,other,5000.,width_s=600.,search_s=2.,bin_s=.01)
        self.assertFalse(row['timing_signal_supported'])
        row['center_hours']=1.
        with self.assertRaises(ValueError):
            fit_clock(pd.DataFrame([row,row,row]),pd.DataFrame([row]))


if __name__=='__main__':
    unittest.main()
