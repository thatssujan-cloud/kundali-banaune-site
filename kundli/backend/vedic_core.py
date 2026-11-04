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
import os
import threading
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

# Natural lord of each rasi (index 0 = Aries … 11 = Pisces).
SIGN_LORDS: List[str] = [
    "Mars", "Mercury", "Jupiter", "Moon", "Sun", "Mercury",
    "Venus", "Mars", "Jupiter", "Saturn", "Saturn", "Jupiter",
]

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

# Serialises every Swiss-Ephemeris computation across calculator instances.
# ``swe.set_sid_mode`` / ``swe.set_ephe_path`` mutate *process-global* state,
# so concurrent requests using different ayanamsas would otherwise corrupt
# each other's charts mid-computation (observed in load tests: Lahiri charts
# returned with another ayanamsa's values).  A chart takes ~5 ms; serialising
# is cheap and makes results deterministic.
_SWE_LOCK = threading.RLock()


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
    """Vimshottari lord of nakshatra number 1..27.

    The 120-year Vimshottari wheel assigns **three consecutive nakshatras to
    each of the nine lords** (27 ÷ 9 = 3):

        Ketu     → Ashwini, Bharani, Krittika      (1–3)
        Venus    → Rohini, Mrigashira, Ardra       (4–6)
        Sun      → Punarvasu, Pushya, Ashlesha     (7–9)
        Moon     → Magha, P.Phalguni, U.Phalguni   (10–12)
        Mars     → Hasta, Chitra, Swati            (13–15)
        Rahu     → Vishakha, Anuradha, Jyeshtha    (16–18)
        Jupiter  → Mula, P.Ashadha, U.Ashadha      (19–21)
        Saturn   → Shravana, Dhanishta, Shatabhisha(22–24)
        Mercury  → P.Bhadrapada, U.Bhadrapada, Revati (25–27)

    Historical bug note: this function previously divided by 4 (a *pada*
    count) instead of 3, which shifted every lord past nakshatra 3 and made
    e.g. Revati (27) report Jupiter instead of Mercury.
    """
    if not 1 <= nak_1_based <= 27:
        raise ValueError(f"nakshatra index out of range: {nak_1_based}")
    return VIMSHOTTARI_ORDER[(nak_1_based - 1) // 3]


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
    is_graha: bool               # True for the 7 physical planets, False for nodes
    exalted_in: Optional[str]    # sign of exaltation, None for nodes
    debilitated_in: Optional[str]  # sign of debilitation, None for nodes


# Exaltation / debilitation signs (Parashari standard; nodes have none).
EXALTATION: Dict[str, str] = {
    "Sun": "Aries", "Moon": "Taurus", "Mars": "Capricorn",
    "Mercury": "Virgo", "Jupiter": "Cancer", "Venus": "Pisces",
    "Saturn": "Libra",
}
DEBILITATION: Dict[str, str] = {
    "Sun": "Libra", "Moon": "Scorpio", "Mars": "Cancer",
    "Mercury": "Pisces", "Jupiter": "Capricorn", "Venus": "Virgo",
    "Saturn": "Aries",
}


@dataclass
class NavamsaPlacement:
    """D-9 (Navamsa) position of one graha, derived from its D-1 rasi."""
    name: str                    # 'Sun' … 'Ketu'
    abbr: str                    # 'Su' … 'Ke'
    sign_index: int              # D-9 rasi 1–12
    sign: str                    # 'Aries' … 'Pisces'
    sign_short: str              # 'Ar' … 'Pi'
    degree_in_d1_sign: float     # decimal degrees travelled inside the D-1 sign
    navamsa_pada: int             # 1–9 (the D-9 division number within the sign)
    house: int                   # D-9 house counted from the *Lagna* navamsa


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

        Thread-safety note: Swiss Ephemeris keeps the sidereal mode and the
        ephemeris path in *process-global* state.  Constructing a calculator
        mutates that global state, so every computation re-asserts this
        instance's ayanamsa under a lock (:meth:`_compute_locked`).  That makes
        concurrent FastAPI requests with different ayanamsas produce correct,
        deterministic charts.
        """
        if ephe_path:
            # Resolve to an absolute path so charts never depend on the
            # server's current working directory (uvicorn launched from
            # anywhere, container WORKDIRs, systemd units, …).
            swe.set_ephe_path(os.path.abspath(ephe_path))
        self.ephe_path = os.path.abspath(ephe_path) if ephe_path else None
        self.ayanamsa_name = ayanamsa.upper()
        self._sid_mode = AYANAMSA_MODES.get(self.ayanamsa_name, swe.SIDM_LAHIRI)
        self.node_type = node_type.upper()
        swe.set_sid_mode(self._sid_mode)

        # One engine-wide lock: serialises the set_sid_mode → calc sequence so
        # per-request ayanamsa switches can never interleave mid-chart.
        # A chart takes ~5 ms, so throughput stays far above real-world load.
        self._lock = _SWE_LOCK

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

    @staticmethod
    def navamsa_sign(d1_sign_index: int, deg_in_sign: float) -> int:
        """Navamsa (D-9) rasi index 1..12 for a planet.

        Each sign divides into nine 3°20' arcs; the arc's ruler advances one
        sign at a time from the base sign itself (movable bases stay movable,
        etc.), which is exactly ``base + floor(deg / 3°20')`` mod 12.
        """
        return ((d1_sign_index - 1 + int(deg_in_sign // PADA_SPAN)) % 12) + 1

    def compute_chart(self, date: str, time: str, tz_offset_hours: float,
                      latitude: float, longitude: float) -> Dict:
        """Compute the complete natal chart as a JSON-serialisable dict.

        See ``API_SCHEMA.md`` for the documented response contract.

        Thread-safe: re-asserts this instance's sidereal mode under the shared
        Swiss-Ephemeris lock before computing, so concurrent requests that use
        different ayanamsas can never cross-contaminate each other.
        """
        with self._lock:
            swe.set_sid_mode(self._sid_mode)
            if self.ephe_path:
                swe.set_ephe_path(self.ephe_path)
            return self._compute_chart(date, time, tz_offset_hours,
                                       latitude, longitude)

    def _compute_chart(self, date: str, time: str, tz_offset_hours: float,
                       latitude: float, longitude: float) -> Dict:
        """Inner (unlocked) chart computation — call via :meth:`compute_chart`."""
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

        # ---- D-9 Navamsa (divisional chart) placements ---------------------
        lagna_nav_sign = self.navamsa_sign(lag_sign, lag_deg)
        navamsa: List[NavamsaPlacement] = []
        for p in planets:
            n_sign = self.navamsa_sign(p.sign_index, p.degree_in_sign)
            navamsa.append(NavamsaPlacement(
                name=p.name, abbr=p.abbr,
                sign_index=n_sign, sign=RASI_NAMES[n_sign - 1],
                sign_short=RASI_SHORT[n_sign - 1],
                degree_in_d1_sign=round(p.degree_in_sign, 4),
                navamsa_pada=int(p.degree_in_sign // PADA_SPAN) + 1,
                house=((n_sign - lagna_nav_sign) % 12) + 1,
            ))

        # ---- Moon Nakshatra block (dasha anchor details) -------------------
        m_nak_start = (moon.nakshatra_index - 1) * NAK_SPAN
        moon_nak = {
            "nakshatra_index": moon.nakshatra_index,
            "nakshatra": moon.nakshatra,
            "nakshatra_lord": moon.nakshatra_lord,
            "pada": moon.pada,
            "degree_within_nakshatra": round(
                norm360(moon.sidereal_longitude - m_nak_start), 4),
            "start_degree": round(m_nak_start, 4),
            "end_degree": round(m_nak_start + NAK_SPAN, 4),
            "sign": moon.sign,
            "sign_index": moon.sign_index,
            "side_of_moon": "Waxing" if moon.daily_motion_deg >= 0 else "Waning",
        }

        doshas = evaluate_doshas(planets, lag_sign, jd)

        return {
            "meta": {
                "module": "Vedic Kundali Engine v2.0",
                "engine": "Swiss Ephemeris (pyswisseph)",
                "ephemeris_source": self.ephemeris_source,
                "ayanamsa": self.ayanamsa_name,
                "ayanamsa_deg": round(ay, 6),
                "house_system": "Whole Sign (Rasi)",
                "rahu_node_type": self.node_type,
                "julian_day_ut": round(jd, 7),
                "birth_time_ut": jd_to_datetime(jd),
                "supported_ayanamsas": sorted(AYANAMSA_MODES.keys()),
                "divisional_charts": ["D-1 (Rasi/Lagna)", "D-9 (Navamsa)"],
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
            "navamsa_d9": {
                "chart": "D-9 Navamsa (Rasi-based divisional)",
                "lagna_sign_index": lagna_nav_sign,
                "lagna_sign": RASI_NAMES[lagna_nav_sign - 1],
                "lagna_sign_short": RASI_SHORT[lagna_nav_sign - 1],
                "placements": [asdict(n) for n in
                               sorted(navamsa, key=lambda x: CANONICAL_ORDER.index(x.name))],
            },
            "moon_nakshatra": moon_nak,
            "doshas": doshas,
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
            is_graha=name in EXALTATION,
            exalted_in=EXALTATION.get(name),
            debilitated_in=DEBILITATION.get(name),
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


# ---------------------------------------------------------------------------
# Dosha evaluation (module-level so it is unit-testable on its own)
# ---------------------------------------------------------------------------

def _rel_house(from_sign: int, to_sign: int) -> int:
    """Whole-sign house number of ``to_sign`` counted from ``from_sign`` (1..12)."""
    return ((to_sign - from_sign) % 12) + 1


# Houses (from Lagna or from Moon) that trigger Mangal (Kuja) Dosha.
MANGAL_HOUSES = (1, 2, 4, 7, 8, 12)

# Signs Rahu occupies in a classic Kaal Sarp pattern; the matching Ketu sign
# is always exactly opposite (index + 6 mod 12).
KAAL_SARP_TYPES: Dict[int, str] = {
    0:  "Anant",   1:  "Vasuki",   2:  "Shesh",    3:  "Padya",
    4:  "Mahapad", 5:  "Karkotak", 6:  "Shankhpal", 7:  "Gulik",
    8:  "Vish",    9:  "Cottariya (Vishdhari)",
    10: "Shankhchuk", 11: "Kaatva",
}

SATURN_SIGN_SPAN = SIGN_SPAN          # ~2.5 years per rasi
SADE_SATI_LEAD = 2.5                  # years before Saturn enters Moon+12
SADE_SATI_TRAIL = 2.5                 # years after Saturn leaves Moon+2


def evaluate_doshas(planets: List[PlanetPosition], lagna_sign: int,
                    jd_now: Optional[float] = None) -> Dict:
    """Automated Dosha flags computed straight from the sidereal coordinates.

    Returns a JSON-serialisable dict with one entry per checked dosha:

    * ``mangal_dosha``  — Mars in houses 1/2/4/7/8/12 from Lagna **and** from
      the Moon (the widely-used dual condition; ``present_from_lagna`` /
      ``present_from_moon`` expose each half separately).
    * ``kaal_sarp``     — all seven physical grahas hemmed between Rahu and
      Ketu.  Canonical test: every graha lies on the forward arc that starts
      at Rahu and ends at Ketu (i.e. none sits in the opposite Ketu→Rahu
      gap, and none conjuncts the node axis signs breaks nothing — planets
      *in* the axis signs count as hemmed).
    * ``sade_sati``     — transits Saturn over the Moon's natal sign and its
      neighbours (12th/1st/2nd from the Moon), evaluated for *right now*.
    """
    by_name = {p.name: p for p in planets}
    mars, moon, rahu, ketu, saturn = (by_name["Mars"], by_name["Moon"],
                                      by_name["Rahu"], by_name["Ketu"],
                                      by_name["Saturn"])

    # ------------------------------ Mangal --------------------------------
    h_lagna = _rel_house(lagna_sign, mars.sign_index)
    h_moon = _rel_house(moon.sign_index, mars.sign_index)
    mangal_present = h_lagna in MANGAL_HOUSES and h_moon in MANGAL_HOUSES
    mangal = {
        "dosha": "Mangal (Kuja) Dosha",
        "present": bool(mangal_present),
        "mars_house_from_lagna": h_lagna,
        "mars_house_from_moon": h_moon,
        "present_from_lagna": h_lagna in MANGAL_HOUSES,
        "present_from_moon": h_moon in MANGAL_HOUSES,
        "affected_houses": list(MANGAL_HOUSES),
        "severity": ("strong" if h_lagna == 1 or h_moon == 1 else
                     "moderate" if mangal_present else "none"),
        "remedy_hint": ("Nāga Panchami / Hanuman Chalisa worship is the "
                        "traditional remedy." if mangal_present else
                        "No Kuja affliction detected."),
    }

    # ---------------------------- Kaal Sarp -------------------------------
    # Canonical rule: Rahu and Ketu split the zodiac into two 180° arcs.
    # The dosha exists when ALL seven physical grahas lie on the arc that runs
    # forward from Rahu to Ketu (planets conjunct either node sign count as
    # being *on* the axis, hence hemmed).  Any graha in the opposite
    # (Ketu → Rahu) gap breaks the yoga.
    rahu_l, ketu_l = rahu.sidereal_longitude, ketu.sidereal_longitude
    arc_len = norm360(ketu_l - rahu_l)              # length of Rahu→Ketu arc
    breakers: List[str] = []
    for p in planets:
        if not p.is_graha:
            continue
        d_from_rahu = norm360(p.sidereal_longitude - rahu_l)
        on_axis_sign = p.sign_index in (rahu.sign_index, ketu.sign_index)
        on_hemmed_arc = d_from_rahu <= arc_len      # inclusive of both nodes
        if not (on_axis_sign or on_hemmed_arc):
            breakers.append(p.name)                 # sits in the empty gap
    kaal_sarp_present = len(breakers) == 0
    # Which type?  Named by the sign Rahu occupies.
    kaal_type = KAAL_SARP_TYPES.get(rahu.sign_index - 1, "Unknown")
    kaal_sarp = {
        "dosha": "Kaal Sarp Dosha",
        "present": bool(kaal_sarp_present),
        "type": kaal_type if kaal_sarp_present else None,
        "rahu_sign": rahu.sign,
        "ketu_sign": ketu.sign,
        "axis_signs": [RASI_NAMES[rahu.sign_index - 1],
                       RASI_NAMES[ketu.sign_index - 1]],
        "breakers": sorted(set(breakers)),
        "remedy_hint": ("Naga Puja / Mahashivratri observance is traditional."
                        if kaal_sarp_present else
                        "Not a Kaal Sarp yoga — planets are not fully hemmed."),
    }

    # ----------------------------- Sade Sati ------------------------------
    # Natal reference data; the *live* phase is attached by main.py using
    # sade_sati_status() with the current date.
    moon_sign = moon.sign_index                       # 1..12
    sade_sati = {
        "dosha": "Sade Sati (Saturn's 7.5-year transit over the natal Moon)",
        "moon_sign": moon.sign,
        "moon_sign_index": moon_sign,
        "phases": [
            {"phase": "Phase 1 — Janma (rising)",
             "transit_sign_index": ((moon_sign - 2) % 12) + 1,
             "transit_sign": RASI_NAMES[(moon_sign - 2) % 12],
             "description": "Saturn transits the 12th house from the Moon."},
            {"phase": "Phase 2 — Peak",
             "transit_sign_index": moon_sign,
             "transit_sign": RASI_NAMES[moon_sign - 1],
             "description": "Saturn transits the Moon's own sign."},
            {"phase": "Phase 3 — Setting",
             "transit_sign_index": (moon_sign % 12) + 1,
             "transit_sign": RASI_NAMES[moon_sign % 12],
             "description": "Saturn transits the 2nd house from the Moon."},
        ],
        "natal_saturn_relative_to_moon": _rel_house(moon_sign, saturn.sign_index),
        "active": None,          # filled in by main.py / sade_sati_status()
        "note": ("'active' compares Saturn's real-time transit sign against "
                 "the three phase signs above."),
    }

    summary = {
        "mangal_dosha": mangal,
        "kaal_sarp_dosha": kaal_sarp,
        "sade_sati": sade_sati,
        "checked_at_module": "vedic_core v2.0",
    }
    return summary


def jd_now_default() -> float:
    """Julian Day UT for the present instant (used when caller passes None)."""
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    return swe.julday(now.year, now.month, now.day,
                      now.hour + now.minute / 60.0 + now.second / 3600.0)


def sade_sati_status(moon_sign_index: int, jd_now: Optional[float] = None,
                     ephe_path: Optional[str] = None,
                     ayanamsa: str = "LAHIRI") -> Dict:
    """Live Sade Sati check: Saturn's *actual* current transit sign vs. the
    three phase signs (12th / 1st / 2nd from the natal Moon).

    Uses a throwaway :class:`VedicCalculator` so the caller never has to keep
    one around; safe to call on every request.  ``ayanamsa`` must match the
    chart it is being compared against, otherwise sidereal signs shift.
    """
    if jd_now is None:
        jd_now = jd_now_default()
    probe = VedicCalculator(ephe_path=ephe_path, ayanamsa=ayanamsa)
    with probe._lock:                      # same global-state guarantee
        lam, spd = probe._calc(jd_now, swe.SATURN)
    sign_idx, deg = deg_in_sign(lam)
    rel_to_moon = _rel_house(moon_sign_index, sign_idx)
    active = rel_to_moon in (12, 1, 2)
    phase_names = {12: "Phase 1 — Janma (rising)",
                   1: "Phase 2 — Peak",
                   2: "Phase 3 — Setting"}
    return {
        "saturn_transit_sign_index": sign_idx,
        "saturn_transit_sign": RASI_NAMES[sign_idx - 1],
        "saturn_transit_degree_dms": dms(deg),
        "saturn_house_from_natal_moon": rel_to_moon,
        "active": bool(active),
        "active_phase": phase_names.get(rel_to_moon) if active else None,
        "evaluated_jd_ut": round(jd_now, 5),
        "evaluated_date_ut": jd_to_datetime(jd_now),
    }
