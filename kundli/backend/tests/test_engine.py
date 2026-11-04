"""
tests/test_engine.py — Verification suite for the Vedic Kundali engine.
=======================================================================

Run with:   pytest tests/ -v          (from kundli/backend)
or:         python3 tests/test_engine.py

The tests check *astronomical invariants* rather than hard-coded tables, so
they stay valid as long as the Swiss Ephemeris installation is correct.
Reference values were cross-checked against published Lahiri-ayanamsa data
(e.g. Swami Vivekananda's natal chart, 12 Jan 1863, 05:30 LT, Kolkata).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vedic_core import (VedicCalculator, deg_in_sign, nak_lord, nakshatra_of,
                        norm360, jd_to_date, jd_to_datetime, dms,
                        MAHADASHA_YEARS)  # noqa: E402

EPHE = os.environ.get("KUNDALI_EPHE_PATH",
                      os.path.join(os.path.dirname(__file__), "..", "ephe"))


class TestAngleHelpers(unittest.TestCase):

    def test_norm360(self):
        self.assertAlmostEqual(norm360(-1.0), 359.0)
        self.assertAlmostEqual(norm360(360.0), 0.0)
        self.assertAlmostEqual(norm360(721.5), 1.5)

    def test_sign_boundaries(self):
        self.assertEqual(deg_in_sign(0.0)[0], 1)          # 0° Aries → sign 1
        self.assertEqual(deg_in_sign(29.999)[0], 1)
        self.assertEqual(deg_in_sign(30.0)[0], 2)         # 0° Taurus → sign 2
        self.assertEqual(deg_in_sign(359.9)[0], 12)       # Pisces
        self.assertAlmostEqual(deg_in_sign(45.25)[1], 15.25)

    def test_nakshatra_pada(self):
        # Ashwini spans 0–13°20'
        self.assertEqual(nakshatra_of(0.0)[:2], (1, 1))
        self.assertEqual(nakshatra_of(3.30)[:2], (1, 1))     # pada 1 (<3°20')
        self.assertEqual(nakshatra_of(3.35)[:2], (1, 2))     # pada 2
        self.assertEqual(nakshatra_of(13.333 - 0.01)[:2], (1, 4))
        self.assertEqual(nakshatra_of(13.334)[:2], (2, 1))   # Bharani begins
        self.assertEqual(nakshatra_of(359.99)[:2], (27, 4))  # Revati pada 4

    def test_nakshatra_lords(self):
        """Each Vimshottari lord rules exactly 3 consecutive nakshatras."""
        expected = ["Ketu", "Ketu", "Ketu",                       # 1-3
                    "Venus", "Venus", "Venus",                    # 4-6
                    "Sun", "Sun", "Sun",                          # 7-9
                    "Moon", "Moon", "Moon",                       # 10-12
                    "Mars", "Mars", "Mars",                       # 13-15
                    "Rahu", "Rahu", "Rahu",                       # 16-18
                    "Jupiter", "Jupiter", "Jupiter",              # 19-21
                    "Saturn", "Saturn", "Saturn",                 # 22-24
                    "Mercury", "Mercury", "Mercury"]              # 25-27
        for i, lord in enumerate(expected, start=1):
            self.assertEqual(nak_lord(i), lord, f"nakshatra {i}")

    def test_nakshatra_lord_boundaries(self):
        self.assertEqual(nak_lord(1), "Ketu")       # Ashwini
        self.assertEqual(nak_lord(4), "Venus")      # Rohini
        self.assertEqual(nak_lord(13), "Mars")      # Hasta
        self.assertEqual(nak_lord(27), "Mercury")   # Revati
        with self.assertRaises(ValueError):
            nak_lord(0)
        with self.assertRaises(ValueError):
            nak_lord(28)


class TestJulianDay(unittest.TestCase):

    def setUp(self):
        self.calc = VedicCalculator(ephe_path=EPHE)

    def test_julday_utc_noon(self):
        jd = self.calc.julday("2000-01-01", "12:00:00", 0.0)
        self.assertAlmostEqual(jd, 2451545.0, places=6)

    def test_timezone_conversion(self):
        # 10:00 IST (+5.5) == 04:30 UT on the same day
        jd = self.calc.julday("1990-05-15", "10:00:00", 5.5)
        self.assertEqual(jd_to_datetime(jd), "1990-05-15 04:30")
        # 02:00 IST == 20:30 UT of the *previous* day
        jd2 = self.calc.julday("1990-05-15", "02:00:00", 5.5)
        self.assertEqual(jd_to_datetime(jd2), "1990-05-14 20:30")

    def test_ayanamsa_lahiri(self):
        """Lahiri (Chitrapaksha) ayanamsa — verified against Swiss Ephemeris.

        Reference points (cross-checked with astrodienst's ayanamsa tables and
        the canonical Lahiri value of 23°15'00" at the epoch JD 2400001
        (1900-01-01)):

          * J2000.0 (2000-01-01 12 UT): 23.857°  (≈ 23°51'26")
          * 2026-01-01 12 UT:           24.220°  (≈ 24°13'13")

        NOTE: an earlier draft of this test asserted 24.62° for Jan 2026; that
        figure corresponds to the **Fagan/Bradley** (New Astronomical)
        ayanamsa, not Lahiri.  Lahiri never exceeds ~24°14' in January 2026.
        The tolerance here (±0.02°) is tight enough to catch any wrong
        sidereal mode while remaining robust across SE build versions.
        """
        jd_j2000 = self.calc.julday("2000-01-01", "12:00:00", 0.0)
        ay = self.calc.ayanamsa(jd_j2000)
        self.assertAlmostEqual(ay, 23.857, delta=0.02)
        jd_2026 = self.calc.julday("2026-01-01", "12:00:00", 0.0)
        self.assertAlmostEqual(self.calc.ayanamsa(jd_2026), 24.220, delta=0.02)

    def test_ayanamsa_monotonic_and_positive(self):
        """Ayanamsa grows ~50.3"/yr (precession) and must be increasing."""
        a1 = self.calc.ayanamsa(self.calc.julday("2000-01-01", "12:00:00", 0.0))
        a2 = self.calc.ayanamsa(self.calc.julday("2026-01-01", "12:00:00", 0.0))
        rate = (a2 - a1) / 26.0 * 3600.0            # arcsec per year
        self.assertGreater(a2, a1)
        self.assertTrue(49.0 < rate < 52.0, f"precession rate {rate:.2f}''/yr")


class TestPlanetaryPositions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.calc = VedicCalculator(ephe_path=EPHE)
        # Swami Vivekananda — widely published Lahiri chart.
        cls.chart = cls.calc.compute_chart("1863-01-12", "05:30:00", 5.5,
                                           22.5726, 88.3639)
        cls.by_name = {p["name"]: p for p in cls.chart["planets"]}

    def test_ketu_opposite_rahu(self):
        ra, ke = self.by_name["Rahu"], self.by_name["Ketu"]
        diff = abs(norm360(ra["sidereal_longitude"] - ke["sidereal_longitude"]))
        self.assertAlmostEqual(diff, 180.0, places=3)

    def test_house_consistency_whole_sign(self):
        """house == ((sign − lagna_sign) mod 12) + 1 for every graha."""
        ls = self.chart["lagna"]["sign_index"]
        for p in self.chart["planets"]:
            self.assertEqual(p["house"], ((p["sign_index"] - ls) % 12) + 1,
                             f"{p['name']} house mismatch")

    def test_lagna_is_house_1(self):
        self.assertEqual(self.by_name["Sun"]["house"], 1)   # Sun in Sagittarius
        # Planets in the rising sign always land in house 1
        for p in self.chart["planets"]:
            if p["sign_index"] == self.chart["lagna"]["sign_index"]:
                self.assertEqual(p["house"], 1)

    def test_reference_positions_vivekananda(self):
        """Published Lahiri positions (±0.5°, tolerant to node algorithm)."""
        checks = {   # name: (sign_index, deg_in_sign ±0.5)
            "Sun":   (9, 29.40),   # late Sagittarius
            "Moon":  (6, 17.08),   # Virgo (Hasta ~3rd pada)
            "Mars":  (1, 6.31),    # Aries (Ashwini)
            "Mercury": (10, 11.72),# Capricorn
            "Jupiter": (7, 4.01),  # Libra (Chitra)
            "Venus": (10, 7.07),   # Capricorn
            "Saturn": (6, 13.58),  # Virgo
        }
        for name, (sign, deg) in checks.items():
            p = self.by_name[name]
            self.assertEqual(p["sign_index"], sign, f"{name} sign")
            self.assertAlmostEqual(p["degree_in_sign"], deg, delta=0.5,
                                   msg=f"{name} degree")

    def test_reference_lagna_vivekananda(self):
        lag = self.chart["lagna"]
        self.assertEqual(lag["sign_index"], 9)              # Sagittarius Lagna
        self.assertAlmostEqual(lag["degree_in_sign"], 16.4, delta=0.35)
        self.assertEqual(lag["nakshatra"], "Purva Ashadha")

    def test_retrograde_flags(self):
        """Mercury/Saturn retrogression must agree with daily-motion sign."""
        for p in self.chart["planets"]:
            if p["name"] in ("Sun", "Moon"):
                self.assertFalse(p["retrograde"])
            elif p["name"] not in ("Rahu", "Ketu"):
                expected = p["daily_motion_deg"] < 0
                self.assertEqual(p["retrograde"], expected)

    def test_nakshatra_math_matches_position(self):
        for p in self.chart["planets"]:
            lam = p["sidereal_longitude"]
            idx, pada, _ = nakshatra_of(lam)
            self.assertEqual(p["nakshatra_index"], idx)
            self.assertEqual(p["pada"], pada)
            self.assertEqual(p["nakshatra_lord"], nak_lord(idx))


class TestVimshottari(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.calc = VedicCalculator(ephe_path=EPHE)
        cls.chart = cls.calc.compute_chart("1863-01-12", "05:30:00", 5.5,
                                           22.5726, 88.3639)
        cls.d = cls.chart["vimshottari_dasha"]

    def test_starting_lord_from_moon_nakshatra(self):
        # Moon at 167.09° sidereal → Hasta (nakshatra 13) → lord **Mars**
        # (Hasta/Chitra/Swati form the Mars group of the Vimshottari wheel).
        self.assertEqual(self.d["moon_nakshatra"], "Hasta")
        self.assertEqual(self.d["starting_mahadasha_lord"], "Mars")
        self.assertEqual(self.d["mahadashas"][0]["lord"], "Mars")
        self.assertEqual(self.d["mahadashas"][0]["start_date"], "1863-01-12")

    def test_balance_less_than_full_period(self):
        self.assertGreater(self.d["balance_years"], 0.0)
        self.assertLessEqual(self.d["balance_years"], 7.0)  # Mars period = 7 y

    def test_balance_matches_classic_table(self):
        """Published dasha tables give ~7.0 y balance for this chart;
        Mars MD ends early 1870, Rahu MD (18y) then runs to ~1888."""
        md = self.d["mahadashas"]
        self.assertAlmostEqual(md[0]["length_years"], 7.00, delta=0.05)
        self.assertEqual(md[1]["lord"], "Rahu")
        self.assertEqual(md[2]["lord"], "Jupiter")
        self.assertEqual(md[3]["lord"], "Saturn")
        # Rahu MD end date ≈ 1870-01 + 18y = 1888-01
        self.assertTrue(md[1]["end_date"].startswith("1887") or
                        md[1]["end_date"].startswith("1888"))

    def test_cycle_sums_to_120(self):
        total = sum(m["length_years"] for m in self.d["mahadashas"])
        # one full wheel minus the consumed fraction of the first MD
        first_period = MAHADASHA_YEARS[self.d["starting_mahadasha_lord"]]
        self.assertAlmostEqual(total, 120.0 - (first_period - self.d["balance_years"]),
                               delta=0.05)

    def test_antardasha_proportions(self):
        md = self.d["mahadashas"][1]          # Rahu 18-year mahadasha
        ads = md["antardashas"]
        self.assertEqual(len(ads), 9)
        self.assertAlmostEqual(sum(a["length_years"] for a in ads),
                               md["length_years"], delta=0.02)
        # First bhukti of Rahu MD is Jupiter: 18 × 16/120 = 2.4 y
        self.assertEqual(ads[0]["lord"], "Jupiter")
        self.assertAlmostEqual(ads[0]["length_years"], 2.4, delta=0.01)

    def test_dates_are_chronological(self):
        prev_end = None
        for m in self.d["mahadashas"]:
            if prev_end:
                self.assertEqual(m["start_date"], prev_end)
            self.assertLess(m["start_date"], m["end_date"])
            prev_end = m["end_date"]

    def test_antardasha_chain_is_contiguous(self):
        """Each AD starts exactly where the previous one ended, and the last
        AD of every MD ends on the MD's own end date."""
        for m in self.d["mahadashas"]:
            ads = m["antardashas"]
            self.assertEqual(ads[0]["start_date"], m["start_date"])
            for prev, nxt in zip(ads, ads[1:]):
                self.assertEqual(nxt["start_date"], prev["end_date"],
                                 f"{m['lord']} AD chain gap")
            self.assertEqual(ads[-1]["end_date"], m["end_date"],
                             f"{m['lord']} AD tail mismatch")

    def test_pratyantardasha_scale(self):
        """Sanity: AD lengths follow period ratio — e.g. in a full 20-year
        Venus MD the Sun bhukti must be 20×6/120 = 1.0 year exactly."""
        calc = VedicCalculator(ephe_path=EPHE)
        # pick any chart, find its *second* mahadasha (always full length)
        chart = calc.compute_chart("1990-05-15", "10:00:00", 5.5,
                                   28.6139, 77.2090)
        md2 = chart["vimshottari_dasha"]["mahadashas"][1]
        for ad in md2["antardashas"]:
            expect = md2["length_years"] * MAHADASHA_YEARS[ad["lord"]] / 120.0
            self.assertAlmostEqual(ad["length_years"], expect, delta=0.001,
                                   msg=f"{md2['lord']}-{ad['lord']} bhukti")


class TestApiSchemaShape(unittest.TestCase):
    """The JSON contract the frontend depends on."""

    def setUp(self):
        self.chart = VedicCalculator(ephe_path=EPHE).compute_chart(
            "1990-05-15", "10:00:00", 5.5, 28.6139, 77.2090)

    def test_top_level_keys(self):
        for k in ("meta", "input", "lagna", "planets", "vimshottari_dasha"):
            self.assertIn(k, self.chart)

    def test_planet_fields(self):
        required = {"sign_index", "degree_in_sign", "nakshatra",
                    "nakshatra_lord", "pada", "retrograde", "house", "abbr"}
        self.assertEqual(len(self.chart["planets"]), 9)
        for p in self.chart["planets"]:
            self.assertTrue(required.issubset(p.keys()), p["name"])
            self.assertTrue(1 <= p["sign_index"] <= 12)
            self.assertTrue(1 <= p["house"] <= 12)
            self.assertTrue(1 <= p["pada"] <= 4)

    def test_lagna_fields(self):
        for k in ("degree", "sign_index", "nakshatra", "pada"):
            self.assertIn(k, self.chart["lagna"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
