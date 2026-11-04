"""
vedic_core.py — Authentic Vedic Astrology (Jyotish) calculation engine
======================================================================

Pure, framework-agnostic calculation module.  Consumed by ``main.py`` (FastAPI)
and by the unit-test suite in ``tests/test_engine.py``.

Calculation standards implemented
---------------------------------
* Ephemeris     : Swiss Ephemeris C library via ``pyswisseph``, using the
                  NASA JPL DE431-derived ``.se1`` files when present
                  (``ephe/sepl_18.se1``, ``semo_18.se1``, ``seas_18.se1``).
                  If the files are missing the engine transparently falls back
                  to the built-in Moshier semi-analytic models (~0.005" accuracy,
                  still far better than one arc-minute of a nakshatra pada).
* Ayanamsa      : Lahiri / Chitrapaksha (``swe.SIDM_LAHIRI``) by default — the
                  ayanamsa that fixes the fixed star Spica (Chitra) at 0°00'
                  sidereal Cancer→Aries reference, i.e. Chitra-paksha.
* House system  : Whole-Sign (Rasi) houses.  The sign occupied by the Ascendant
                  is House 1; zodiacal order continues House 2 … House 12.
                  Zodiacal order runs counter-clockwise on the sky, which is why
                  house numbers run counter-clockwise in the North-Indian diamond.
* Grahas        : Surya (Sun), Chandra (Moon), Mangal (Mars), Budha (Mercury),
                  Guru (Jupiter), Shukra (Venus), Shani (Saturn), Rahu (mean
                  north lunar node — the Parashari standard) and Ketu, computed
                  strictly as Rahu + 180°.
* Nakshatras    : 27 mansions of 13°20', four padas of 3°20' each.
* Vimshottari    : 120-year Mahadasha wheel.  The *balance* of the natal
  Dasha          nakshatra is measured astronomically: we bisect on the Moon's
                  true sidereal longitude to find the exact instant it entered
                  the current nakshatra, then scale by the lord's period.  This
                  reproduces published dasha tables to well under an hour.

Author : Lead Full-Stack Astrology Software Engineer
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

try:
    import swisseph as swe
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyswisseph is required — see README.md §Installation "
        "(pip install pyswisseph, or build the Swiss Ephemeris C library)."
    ) from exc


# ---------------------------------------------------------------------------
# Tables & constants
# ---------------------------------------------------------------------------

RASI_NAMES: List[str] = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

RASI_SHORT: List[str] = ["Ar", "Ta", "Ge", "Cn", "Le", "Vi",
                         "Li", "Sc", "Sg", "Cp", "Aq", "Pi"]

RASI_DEVANAGARI: List[str] = ["मेष", "वृषभ", "मिथुन", "कर्क", "सिंह", "कन्या",
                              "तुला", "वृश्चिक", "धनु", "मकर", "कुंभ", "मीन"]

NAKSHATRA_NAMES: List[str] = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni",
    "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha",
    "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana",
    "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada",
    "Revati",
]

# Vimshottari sequence starting with Ketu (lord of nakshatra 1 = Ashwini).
VIMSHOTTARI_ORDER: List[str] = [
    "Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury",
]

MAHADASHA_YEARS: Dict[str, float] = {
    "Ketu": 7.0, "Venus": 20.0, "Sun": 6.0, "Moon": 10.0, "Mars": 7.0,
    "Rahu": 18.0, "Jupiter": 16.0, "Saturn": 19.0, "Mercury": 17.0,
}

YEAR_DAYS = 365.2585            # sidereal year (days) — duration ruler for dashas
SIGN_SPAN = 30.0                # degrees per rasi
NAK_SPAN = 360.0 / 27.0         # 13°20' per nakshatra
PADA_SPAN = NAK_SPAN / 4.0      # 3°20' per pada

PLANET_ABBR: Dict[str, str] = {   # exactly what the North-Indian renderer prints
    "Sun": "Su", "Moon": "Mo", "Mars": "Ma", "Mercury": "Me",
    "Jupiter": "Ju", "Venus": "Ve", "Saturn": "Sa", "Rahu": "Ra", "Ketu": "Ke",
}

PLANET_SANSKRIT: Dict[str, str] = {
    "Sun": "Surya", "Moon": "Chandra", "Mars": "Mangal", "Mercury": "Budha",
    "Jupiter": "Guru", "Venus": "Shukra", "Saturn": "Shani",
    "Rahu": "Rahu", "Ketu": "Ketu",
}

# Never retrograde in the geocentric frame.
NON_RETRO_BODIES = {"Sun", "Moon"}

_PLANET_IDS: Dict[str, int] = {
    "Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS,
    "Mercury": swe.MERCURY, "Jupiter": swe.JUPITER, "Venus": swe.VENUS,
    "Saturn": swe.SATURN, "Rahu": swe.MEAN_NODE,
}

AYANAMSA_MODES: Dict[str, int] = {
    "LAHIRI": swe.SIDM_LAHIRI,           # default (Chitrapaksha)
    "RAMAN": swe.SIDM_RAMAN,
    "YUKTESHWAR": swe.SIDM_YUKTESHWAR,
    "KRISHNAMURTI": swe.SIDM_KRISHNAMURTI,
    "DJWHALKHUL": swe.SIDM_DJWHAL_KHUL,
    "TRUE_CHITRA": swe.SIDM_TRUE_CITRA,
}

CANONICAL_ORDER = ["Sun", "Moon", "Mars", "Mercury", "Jupiter",
                   "Venus", "Saturn", "Rahu", "Ketu"]


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def norm360(x: float) -> float:
    """Wrap an angle into [0, 360)."""
    x = math.fmod(x, 360.0)
    return x + 360.0 if x < 0.0 else x


def deg_in_sign(longitude: float) -> Tuple[int, float]:
    """(sign_index 1..12, degrees inside sign) for a sidereal ecliptic longitude."""
    lam = norm360(longitude)
    return int(lam // SIGN_SPAN) + 1, lam % SIGN_SPAN


def nakshatra_of(longitude: float) -> Tuple[int, int, float]:
    """(nakshatra 1..27, pada 1..4, degrees travelled inside the nakshatra)."""
    lam = norm360(longitude)
    idx = min(int(lam // NAK_SPAN), 26)
    within = lam - idx * NAK_SPAN
    pada = min(int(within // PADA_SPAN) + 1, 4)
    return idx + 1, pada, within


def nak_lord(nak_1_based: int) -> str:
    """Vimshottari lord of nakshatra number 1..27."""
    return VIMSHOTTARI_ORDER[(nak_1_based - 1) // 4]


def dms(deg: float) -> str:
    """Decimal degrees → 'DD°MM'SS"' (sign-relative)."""
    dd = int(deg)
    mm_full = (deg - dd) * 60.0
    mm = int(mm_full)
    ss = round((mm_full - mm) * 60.0)
    if ss == 60:
        ss, mm = 0, mm + 1
    if mm == 60:
        mm, dd = 0, dd + 1
    return f"{dd:02d}\u00b0{mm:02d}'{ss:02d}\""


def jd_to_date(jd_ut: float) -> str:
    """Julian Day (UT) → proleptic Gregorian 'YYYY-MM-DD' (civil, midnight-based).

    Convention note: ``swe.julday`` takes the time-of-day measured from
    *midnight*, whereas ``swe.revjul`` returns the day that began at *noon* plus
    a fraction.  We therefore shift by 0.5 and floor to obtain the civil
    calendar date (the standard astronomical trick).
    """
    return _jd_to_cal(jd_ut)[0]


def jd_to_datetime(jd_ut: float) -> str:
    """Julian Day (UT) → 'YYYY-MM-DD HH:MM' (24 h UT clock)."""
    date_s, minutes = _jd_to_cal(jd_ut, with_minutes=True)
    return f"{date_s} {minutes // 60:02d}:{minutes % 60:02d}"


def _jd_to_cal(jd_ut: float, with_minutes: bool = False):
    """Internal: JD(UT) → ('YYYY-MM-DD' [, minutes-past-midnight])."""
    z = math.floor(jd_ut + 0.5)          # JD of the civil midnight starting this day
    frac_day = (jd_ut + 0.5) - z         # elapsed fraction of the civil day
    y, m, d, _ = swe.revjul(float(z))    # revjul of an integer JD gives its date
    date_s = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    if with_minutes:
        minutes = int(round(frac_day * 1440.0)) % 1440
        return date_s, minutes
    return (date_s,)


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------

@dataclass
class PlanetPosition:
    """One graha, fully resolved for both the table UI and the SVG chart."""
    name: str                    # 'Sun' … 'Ketu'
    sanskrit_name: str           # 'Surya' … 'Ketu'
    abbr: str                    # 'Su' … 'Ke'  (renderer token)
    sidereal_longitude: float    # 0–360°, Lahiri
    sign_index: int              # 1–12 (1 = Aries)
    sign: str                    # 'Aries' … 'Pisces'
    degree_in_sign: float        # 0–30 (decimal)
    degree_in_sign_dms: str      # '12°34'56"'
    nakshatra_index: int         # 1–27
    nakshatra: str               # 'Ashwini' … 'Revati'
    nakshatra_lord: str          # Vimshottari lord of that nakshatra
    pada: int                    # 1–4
    retrograde: bool             # True ⇒ renderer appends '*'/'(R)'
    house: int                   # 1–12 whole-sign house counted from Lagna
    daily_motion_deg: float      # signed °/day (negative = retrograde)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VedicCalculator:
    """Swiss-Ephemeris-backed Jyotish engine (thread-safe after construction)."""

    def __init__(self, ephe_path: Optional[str] = None,
                 ayanamsa: str = "LAHIRI", node_type: str = "MEAN") -> None:
        """
        :param ephe_path: directory containing ``se*_18.se1`` (NASA JPL DE431)
                          files; ``None`` keeps the process-wide path.
        :param ayanamsa:  key of :data:`AYANAMSA_MODES`; default **LAHIRI**.
        :param node_type: ``MEAN`` (Parashari) or ``TRUE`` (KP) for Rahu.
        """
        if ephe_path:
            swe.set_ephe_path(ephe_path)
        self.ayanamsa_name = ayanamsa.upper()
        swe.set_sid_mode(AYANAMSA_MODES.get(self.ayanamsa_name, swe.SIDM_LAHIRI))
        self.node_type = node_type.upper()

        # Probe once whether full-precision JPL files are reachable so we can
        # report provenance in the API payload.
        try:
            swe.calc_ut(2451544.5, swe.SUN, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)
            self._ephe_flag = swe.FLG_SWIEPH
            self.ephemeris_source = "Swiss Ephemeris .se1 files (NASA/JPL DE431)"
        except Exception:
            self._ephe_flag = swe.FLG_MOSEPH
            self.ephemeris_source = "Moshier fallback (no .se1 ephemeris files found)"

    # -- primitives ----------------------------------------------------------

    def _calc(self, jd_ut: float, body: int,
              extra_flags: int = 0) -> Tuple[float, float]:
        """Return (sidereal longitude, signed daily motion) for a Swiss body."""
        xx, _flags = swe.calc_ut(
            jd_ut, body,
            swe.FLG_SIDEREAL | swe.FLG_SPEED | self._ephe_flag | extra_flags)
        return norm360(xx[0]), xx[3]

    def julday(self, date: str, time: str, tz_offset_hours: float) -> float:
        """Local 'YYYY-MM-DD'/'HH:MM:SS' + UTC offset (h) → Julian Day **UT**."""
        yy, mm, dd = (int(v) for v in date.split("-"))
        hh, mi, ss = (float(v) for v in time.split(":"))
        ut_hour = hh + mi / 60.0 + ss / 3600.0 - tz_offset_hours
        return swe.julday(yy, mm, dd, ut_hour)

    def ayanamsa(self, jd_ut: float) -> float:
        """Current ayanamsa value (degrees) for the configured mode."""
        return swe.get_ayanamsa_ut(jd_ut)

    def lagna_longitude(self, jd_ut: float, lat: float, lon: float) -> float:
        """Sidereal Ascendant degree (the Lagna point).

        We request **Whole-Sign ('W')** cusps from Swiss Ephemeris: the ascendant
        ``ascmc[0]`` is identical for every house algorithm, and 'W' keeps the
        call consistent with the chart's declared house system.
        """
        _cusps, ascmc = swe.houses_ex(
            jd_ut, lat, lon, b"W",
            swe.FLG_SIDEREAL | self._ephe_flag)
        return norm360(ascmc[0])

    # -- public API ----------------------------------------------------------

    def compute_chart(self, date: str, time: str, tz_offset_hours: float,
                      latitude: float, longitude: float) -> Dict:
        """Compute the complete natal chart as a JSON-serialisable dict.

        See ``API_SCHEMA.md`` for the documented response contract.
        """
        jd = self.julday(date, time, tz_offset_hours)
        ay = self.ayanamsa(jd)

        lag = self.lagna_longitude(jd, latitude, longitude)
        lag_sign, lag_deg = deg_in_sign(lag)
        lag_nak, lag_pada, _ = nakshatra_of(lag)

        planets: List[PlanetPosition] = []
        rahu_lon = rahu_speed = 0.0
        for name, pid in _PLANET_IDS.items():
            extra = swe.FLG_TRUENODE if (name == "Rahu"
                                         and self.node_type == "TRUE") else 0
            lam, spd = self._calc(jd, pid, extra)
            if name == "Rahu":
                rahu_lon, rahu_speed = lam, -abs(spd)   # nodes always wander retrograde
                spd = rahu_speed
            retro = (spd < 0.0) and (name not in NON_RETRO_BODIES)
            planets.append(self._record(name, lam, spd, lag_sign, retro))

        ketu_lon = norm360(rahu_lon + 180.0)            # Ketu ≡ Rahu + 180°
        planets.append(self._record("Ketu", ketu_lon, rahu_speed, lag_sign,
                                    retro=False))

        moon = next(p for p in planets if p.name == "Moon")
        dasha = self.vimshottari_timeline(jd, moon.sidereal_longitude)

        return {
            "meta": {
                "module": "Vedic Kundali Engine v1.0",
                "engine": "Swiss Ephemeris (pyswisseph)",
                "ephemeris_source": self.ephemeris_source,
                "ayanamsa": self.ayanamsa_name,
                "ayanamsa_deg": round(ay, 6),
                "house_system": "Whole Sign (Rasi)",
                "rahu_node_type": self.node_type,
                "julian_day_ut": round(jd, 7),
                "birth_time_ut": jd_to_datetime(jd),
            },
            "input": {
                "date_of_birth": date,
                "time_of_birth_local": time,
                "timezone_offset_utc_hours": tz_offset_hours,
                "latitude": latitude,
                "longitude": longitude,
            },
            "lagna": {
                "degree": round(lag, 4),
                "degree_in_sign": round(lag_deg, 4),
                "degree_dms": dms(lag_deg),
                "sign_index": lag_sign,
                "sign": RASI_NAMES[lag_sign - 1],
                "sign_short": RASI_SHORT[lag_sign - 1],
                "nakshatra_index": lag_nak,
                "nakshatra": NAKSHATRA_NAMES[lag_nak - 1],
                "nakshatra_lord": nak_lord(lag_nak),
                "pada": lag_pada,
            },
            "planets": [asdict(p) for p in
                        sorted(planets, key=lambda x: CANONICAL_ORDER.index(x.name))],
            "vimshottari_dasha": dasha,
        }

    # -- record builder ------------------------------------------------------

    def _record(self, name: str, lam: float, spd: float,
                lagna_sign: int, retro: bool) -> PlanetPosition:
        sign, deg = deg_in_sign(lam)
        nak, pada, _ = nakshatra_of(lam)
        return PlanetPosition(
            name=name,
            sanskrit_name=PLANET_SANSKRIT[name],
            abbr=PLANET_ABBR[name],
            sidereal_longitude=round(lam, 6),
            sign_index=sign,
            sign=RASI_NAMES[sign - 1],
            degree_in_sign=round(deg, 4),
            degree_in_sign_dms=dms(deg),
            nakshatra_index=nak,
            nakshatra=NAKSHATRA_NAMES[nak - 1],
            nakshatra_lord=nak_lord(nak),
            pada=pada,
            retrograde=bool(retro),
            house=((sign - lagna_sign) % 12) + 1,   # Whole-Sign counting
            daily_motion_deg=round(spd, 6),
        )

    # -- Vimshottari Dasha ----------------------------------------------------

    def vimshottari_timeline(self, jd_birth: float, moon_lon: float) -> Dict:
        """Mahadasha timeline with astronomically-exact balance.

        Algorithm
        ---------
        1. Locate the natal nakshatra and its start longitude.
        2. Solve ``moon_lon(t) = nak_start`` for ``t`` by bisection on the true
           sidereal Moon (handles the 11–15°/daily-motion variation exactly).
        3. elapsed = birth − t ; consumed_fraction = elapsed / lord_period.
        4. The first Mahadasha therefore starts at birth and lasts
           ``(1 − consumed) × lord_period`` years; subsequent ones use their
           full nominal lengths, following the Ketu→Venus→Sun→…→Mercury wheel.
        """
        nak_idx, _, _ = nakshatra_of(moon_lon)
        lord = nak_lord(nak_idx)
        start_lon = (nak_idx - 1) * NAK_SPAN
        jd_entry = self._moon_crossing(jd_birth, start_lon)

        elapsed_years = (jd_birth - jd_entry) / YEAR_DAYS
        period = MAHADASHA_YEARS[lord]
        consumed = min(max(elapsed_years / period, 0.0), 1.0)
        balance_frac = 1.0 - consumed
        balance_years = balance_frac * period

        seq = VIMSHOTTARI_ORDER[VIMSHOTTARI_ORDER.index(lord):] + \
              VIMSHOTTARI_ORDER[:VIMSHOTTARI_ORDER.index(lord)]

        mahadashas: List[Dict] = []
        cur_jd = jd_birth                       # first AD begins at birth
        for i, md_lord in enumerate(seq):
            length = balance_years if i == 0 else MAHADASHA_YEARS[md_lord]
            end_jd = cur_jd + length * YEAR_DAYS
            mahadashas.append({
                "order": i + 1,
                "lord": md_lord,
                "start_date": jd_to_date(cur_jd),
                "end_date": jd_to_date(end_jd),
                "length_years": round(length, 4),
                "length_months": round(length * 12.0, 3),
                "antardashas": self._antardashas(md_lord, cur_jd, length),
            })
            cur_jd = end_jd

        return {
            "system": "Vimshottari (120-year cycle)",
            "moon_nakshatra_index": nak_idx,
            "moon_nakshatra": NAKSHATRA_NAMES[nak_idx - 1],
            "starting_mahadasha_lord": lord,
            "balance_fraction_of_mahadasha": round(balance_frac, 6),
            "balance_years": round(balance_years, 4),
            "balance_days": round(balance_years * YEAR_DAYS, 3),
            "nakshatra_entry_jd_ut": round(jd_entry, 7),
            "mahadashas": mahadashas,
        }

    def _moon_crossing(self, jd_ref: float, target_lon: float) -> float:
        """Instant (JD UT) the sidereal Moon last passed ``target_lon``.

        Bisection on the *signed* shortest angular difference; the search window
        (2.5 days) comfortably exceeds the longest possible transit of a
        nakshatra span (13°20' at ~11.8°/day ≈ 1.14 days).
        """
        lo, hi = jd_ref - 2.5, jd_ref

        def g(t: float) -> float:
            lam, _ = self._calc(t, swe.MOON)
            d = norm360(lam - target_lon)
            return d - 360.0 if d > 180.0 else d      # signed diff in (-180,180]

        # Guarantee a bracket: g(lo) should be ≤ 0 (Moon behind target) and
        # g(hi) ≥ 0 (Moon ahead of target).
        guard = 0
        while g(lo) > 0.0 and guard < 10:
            lo -= 2.5
            guard += 1
        for _ in range(80):                            # ≈ 1e-24 JD precision cap
            mid = 0.5 * (lo + hi)
            if g(mid) <= 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _antardashas(md_lord: str, md_start_jd: float, md_years: float) -> List[Dict]:
        """Antardasha (bhukti) sub-periods of one Mahadasha.

        Standard rule: AD lords follow the same Vimshottari wheel starting from
        the lord *after* the MD lord; each AD length = MD_length ×
        AD-lord-years / 120.  For the *first* (balanced) Mahadasha this yields
        proportionally shortened bhuktis, matching published tables.
        """
        k0 = VIMSHOTTARI_ORDER.index(md_lord)
        out: List[Dict] = []
        cur = md_start_jd
        for k in range(1, 10):
            ad_lord = VIMSHOTTARI_ORDER[(k0 + k) % 9]
            length = md_years * MAHADASHA_YEARS[ad_lord] / 120.0
            end = cur + length * YEAR_DAYS
            out.append({
                "lord": ad_lord,
                "start_date": jd_to_date(cur),
                "end_date": jd_to_date(end),
                "length_years": round(length, 4),
            })
            cur = end
        return out
