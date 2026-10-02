"""Meaningful probability mapping, scoring, bin-edge and fold-exclusion tests."""
import unittest
import numpy as np
from calibration import mapped_probabilities,brier_contributions,calibration_bins,check_fold

class ReversedClasses:
    classes_=np.array([2,1])
    def predict_proba(self,frame):return np.tile([.7,.3],(len(frame),1))

class CalibrationTests(unittest.TestCase):
    def test_probability_mapping_uses_fitted_classes(self):
        myocarditis,acs,classes=mapped_probabilities(ReversedClasses(),np.zeros((2,1)))
        np.testing.assert_allclose(myocarditis,[.3,.3]);np.testing.assert_allclose(acs,[.7,.7]);self.assertEqual(classes,[2,1])

    def test_binary_brier_is_not_double_class_error(self):
        result=brier_contributions([.8,.2],[1,0])
        np.testing.assert_allclose(result,[.04,.04]);self.assertAlmostEqual(result.mean(),.04)

    def test_all_exact_bin_boundaries_and_one(self):
        edges=np.linspace(0,1,6)
        table=calibration_bins(edges,[0,0,0,1,1,1],5,'test','myocarditis_probability')
        self.assertEqual(table.patient_count.tolist(),[1,1,1,1,2])
        self.assertAlmostEqual(table.observed_proportion.iloc[-1],1)

    def test_empty_bins_are_missing_not_zero(self):
        table=calibration_bins([0,1],[0,1],5,'test','myocarditis_probability')
        self.assertTrue(table.loc[table.patient_count==0,'observed_proportion'].isna().all())
        self.assertTrue(table.loc[table.patient_count==0,'mean_predicted_probability'].isna().all())

    def test_fold_rejects_held_out_patient_and_duplicates(self):
        check_fold([2,3],1,2)
        with self.assertRaises(ValueError):check_fold([1,3],1,2)
        with self.assertRaises(ValueError):check_fold([2,2],1,2)
        with self.assertRaises(ValueError):check_fold([2],1,2)

    def test_invalid_probability_simplex_rejected(self):
        class Bad(ReversedClasses):
            def predict_proba(self,frame):return np.tile([.7,.6],(len(frame),1))
        with self.assertRaises(ValueError):mapped_probabilities(Bad(),np.zeros((1,1)))

if __name__=='__main__':unittest.main()
