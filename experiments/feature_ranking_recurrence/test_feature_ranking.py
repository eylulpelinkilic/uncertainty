"""Scientific edge-case checks, independent of the clinical artifacts."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_ranking_recurrence import panel_order, summarize

class RankingTests(unittest.TestCase):
    def test_signed_order_and_strict_positive_eligibility(self):
        features = ['negative', 'zero', 'small', 'large']
        values = [-100, 0, .2, 5]
        self.assertEqual(panel_order(values, features, 'uncertainty').index.tolist(), ['large','small','zero','negative'])
        self.assertEqual(panel_order(values, features, 'low_score').index.tolist(), ['small','large'])

    def test_absences_do_not_become_zero_rank(self):
        features = [f'f{i}' for i in range(22)]
        values = np.vstack([np.arange(22), np.arange(22)[::-1]])
        summary, _ = summarize(values, [1,2], features, features, np.arange(22), 'test', 'original')
        row = summary[(summary.panel=='uncertainty') & (summary.feature=='f0')].iloc[0]
        self.assertEqual(row.displayed_rank_denominator, 1)
        self.assertEqual(row.mean_displayed_rank, 1)
        self.assertEqual(row.mean_full_eligible_rank, 11.5)
        low, _ = summarize(np.array([[-1.,0.,2.]]), [1], ['a','b','c'], ['a','b','c'], [0.,0.,.1], 'test','original')
        absent = low[(low.panel=='low_score') & (low.feature=='a')].iloc[0]
        self.assertTrue(np.isnan(absent.mean_displayed_rank))
        self.assertEqual(absent.displayed_rank_denominator,0)

    def test_cap_zero_divergence_and_scaler_cancellation(self):
        original = np.array([[-3.,2.,1.],[1.,-2.,1.],[2.,1.,1.]])
        untouched = original.copy()
        js = np.array([0.,.03,.2]); eps=1e-12; tau=.05
        capped = original.copy() * (js+eps)/(np.maximum(js,tau)+eps)
        self.assertLess(abs(capped[0,0]),abs(original[0,0]))
        np.testing.assert_array_equal(original, untouched)
        np.testing.assert_allclose(StandardScaler().fit_transform(original),
                                   StandardScaler().fit_transform(capped),rtol=1e-10,atol=1e-10)
        self.assertFalse(np.allclose(StandardScaler().fit(original).transform(capped),
                                    StandardScaler().fit_transform(original)))

if __name__=='__main__': unittest.main()
