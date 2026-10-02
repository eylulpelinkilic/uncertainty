"""Scientific edge-case checks, independent of the clinical artifacts."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_ranking_recurrence import panel_order, summarize

class RankingTests(unittest.TestCase):
    def test_stable_ties_cross_fifth_and_twentieth_in_both_panels(self):
        # Nonalphabetic ordered names ensure tie breaking follows configuration, not name.
        features = [f'f{30-i}' for i in range(27)]
        for panel, values in [('uncertainty', [4.,3.,2.,1.] + [0.]*20 + [-1.,-2.,-3.]),
                              ('low_score', [1.,2.,3.,4.] + [5.]*20 + [6.,7.,8.])]:
            order = panel_order(values, features, panel, 'stable').index.tolist()
            self.assertEqual(order, features)
            table, ranks = summarize(np.array([values]), [1], features, features,
                                     np.arange(27), 'test', 'original', 'stable')
            self.assertEqual(ranks.loc[(ranks.panel==panel)&ranks.top_five,'feature'].tolist(), features[:5])
            self.assertEqual(ranks.loc[(ranks.panel==panel)&ranks.displayed,'feature'].tolist(), features[:20])
            absent = table[(table.panel==panel)&(table.feature==features[20])].iloc[0]
            self.assertEqual(absent.displayed_count,0)
            self.assertTrue(np.isnan(absent.mean_displayed_rank))
            self.assertEqual(absent.mean_full_eligible_rank,21)

    def test_top_five_order_is_distinct_from_membership(self):
        from experiments.feature_ranking_recurrence.analysis import compare_sequences
        a = list('abcdefghijklmnopqrstuv')
        b = a.copy(); b[0],b[1]=b[1],b[0]
        comparison=compare_sequences(a,b)
        self.assertFalse(comparison['top_five_membership_changed'])
        self.assertTrue(comparison['top_five_order_changed'])
        self.assertTrue(comparison['displayed_order_changed'])
        b=a.copy();b[19],b[20]=b[20],b[19]
        comparison=compare_sequences(a,b)
        self.assertFalse(comparison['top_five_order_changed'])
        self.assertTrue(comparison['displayed_membership_changed'])

    def test_identical_tie_rule_after_denominator_cap(self):
        features=['last_name','first_name','middle_name','zero']
        scores=np.array([4.,2.,2.,0.]);js=np.array([.1,.2,.2,0.]);eps=1e-12
        factor=(js+eps)/(np.maximum(js,.3)+eps)
        capped=scores*factor
        self.assertEqual(panel_order(capped,features,'uncertainty','stable').index.tolist()[1:3],features[1:3])
        self.assertEqual(panel_order(capped,features,'low_score','stable').index.tolist()[:2],features[1:3])
        np.testing.assert_array_equal(np.sign(capped),np.sign(scores))

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
