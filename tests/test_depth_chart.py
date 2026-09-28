"""
Tests for depth_chart.py: detecting a promoted backup and scaling his volume projection.

Run:  python -m unittest discover -s tests -v
"""

import unittest

import pandas as pd

import depth_chart as dc


def _qb_games(player_id, team, attempts_per_game, n=8, season=2026):
    return [dict(player_id=player_id, team=team, position="QB", season=season, week=w, attempts=a)
            for w, a in zip(range(1, n + 1), attempts_per_game if hasattr(attempts_per_game, "__len__")
                           else [attempts_per_game] * n)]


class TestUsageRates(unittest.TestCase):
    def test_team_usage_rate_averages_across_whoever_played(self):
        rows = _qb_games("starter", "AAA", 34, n=6) + _qb_games("backup", "AAA", 10, n=2, season=2026)
        # backup's 2 games are weeks 7-8 in this construction would collide; keep separate weeks
        rows = _qb_games("starter", "AAA", 34, n=6) + [
            dict(player_id="backup", team="AAA", position="QB", season=2026, week=7, attempts=10),
            dict(player_id="backup", team="AAA", position="QB", season=2026, week=8, attempts=12)]
        rate, n = dc.usage_rate(pd.DataFrame(rows), "AAA", "QB")
        self.assertEqual(n, 8)
        self.assertGreater(rate, 10)   # pulled up by the starter's weeks, not just the backup's

    def test_non_qb_rb_wr_te_position_returns_zero_not_a_crash(self):
        self.assertEqual(dc.usage_rate(pd.DataFrame(_qb_games("x", "AAA", 30)), "AAA", "DEF"), (0.0, 0))

    def test_player_usage_rate_uses_only_that_players_games(self):
        rows = _qb_games("starter", "AAA", 34, n=8)
        rate, n = dc.player_usage_rate(pd.DataFrame(rows), "starter", "QB")
        self.assertAlmostEqual(rate, 34.0, delta=0.5)
        self.assertEqual(n, 8)

    def test_unknown_player_returns_zero(self):
        rows = _qb_games("starter", "AAA", 34, n=8)
        self.assertEqual(dc.player_usage_rate(pd.DataFrame(rows), "nobody", "QB"), (0.0, 0))


class TestPromotion(unittest.TestCase):
    def _league(self):
        # a clear starter (34 attempts/g, 10 games) and a clear backup (6 attempts/g, 4 games)
        return _qb_games("starter", "AAA", 34, n=10) + _qb_games("backup", "AAA", 6, n=4)

    def test_backup_is_promoted_when_the_starter_is_out(self):
        h = pd.DataFrame(self._league())
        self.assertTrue(dc.is_promoted_starter(h, "AAA", "QB", "backup", {"starter": "Out"}))

    def test_backup_is_not_promoted_when_the_starter_is_merely_questionable(self):
        h = pd.DataFrame(self._league())
        self.assertFalse(dc.is_promoted_starter(h, "AAA", "QB", "backup", {"starter": "Questionable"}))

    def test_backup_is_not_promoted_when_the_starter_has_no_injury_listed(self):
        h = pd.DataFrame(self._league())
        self.assertFalse(dc.is_promoted_starter(h, "AAA", "QB", "backup", {}))

    def test_the_starter_himself_is_never_flagged_as_promoted(self):
        h = pd.DataFrame(self._league())
        self.assertFalse(dc.is_promoted_starter(h, "AAA", "QB", "starter", {"backup": "Out"}))

    def test_a_committee_lead_back_is_not_flagged_when_no_rb_is_hurt(self):
        # a healthy 3-way RB committee (55/30/15% of carries) with nobody injured: none of them
        # should be "promoted", even the lead back, whose own share is well under any fixed
        # percent-of-team-usage threshold -- this is the exact bug a real slate caught (a real
        # team's committee lead back and starting WRs were flagged "promoted" with nobody hurt
        # at their position, because the old check used a 70%-of-team-share cutoff)
        def rb_games(pid, carries, n=8):
            return [dict(player_id=pid, team="AAA", position="RB", season=2026, week=w, carries=carries)
                    for w in range(1, n + 1)]
        h = pd.DataFrame(rb_games("lead", 15) + rb_games("change", 8) + rb_games("third", 4))
        for pid in ("lead", "change", "third"):
            self.assertFalse(dc.is_promoted_starter(h, "AAA", "RB", pid, {}),
                            f"{pid} should not be promoted -- nobody at RB is injured")

    def test_a_teammate_with_too_little_history_does_not_block_promotion(self):
        # a third QB has thrown MORE per game but only in 1 game (garbage time) -- shouldn't
        # count as "the healthy starter" and block the real backup's promotion
        h = pd.DataFrame(self._league() + _qb_games("scout_team_arm", "AAA", 40, n=1))
        self.assertTrue(dc.is_promoted_starter(h, "AAA", "QB", "backup", {"starter": "Out"}))

    def test_multiplier_is_neutral_for_the_normal_starter(self):
        h = pd.DataFrame(self._league())
        self.assertEqual(dc.promotion_multiplier(h, "AAA", "QB", "starter", {"backup": "Out"}), 1.0)

    def test_multiplier_scales_up_but_is_capped_and_trust_limited(self):
        h = pd.DataFrame(self._league())
        mult = dc.promotion_multiplier(h, "AAA", "QB", "backup", {"starter": "Out"})
        team_rate, _ = dc.usage_rate(h, "AAA", "QB")
        own_rate, _ = dc.player_usage_rate(h, "backup", "QB")
        full_ratio = team_rate / own_rate
        self.assertGreater(mult, 1.0)
        self.assertLess(mult, full_ratio)              # PROMOTION_TRUST holds back some of the gap
        self.assertLessEqual(mult, 1 + dc.PROMOTION_TRUST * (dc.PROMOTION_CAP - 1))

    def test_multiplier_is_neutral_when_not_promoted(self):
        h = pd.DataFrame(self._league())
        self.assertEqual(dc.promotion_multiplier(h, "AAA", "QB", "backup", {}), 1.0)

    def test_no_history_at_all_is_handled(self):
        h = pd.DataFrame(columns=["player_id", "team", "position", "season", "week", "attempts"])
        self.assertFalse(dc.is_promoted_starter(h, "AAA", "QB", "backup", {"starter": "Out"}))
        self.assertEqual(dc.promotion_multiplier(h, "AAA", "QB", "backup", {"starter": "Out"}), 1.0)


if __name__ == "__main__":
    unittest.main()
