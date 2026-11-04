"""Space environment: Sun, Earth shadow, geomagnetic field, ground stations, atmosphere.

Frames. Basilisk integrates the orbit with point-mass gravity, so its inertial frame 'P' keeps the
orbit plane fixed. Real LEO planes precess under J2 (sun-synchronous orbits by design). We keep the
dynamics in P and rotate every external vector into P:  v_P = Rz(-dOmega(t)) v_ECI, with
dOmega(t) = Omega_dot_J2 * t [VAL eq. 9-41]. The rotation rate (~2e-7 rad/s) is dynamically
negligible; geometry (Sun angle, eclipses, field, passes) is then exactly as for a precessing orbit.

  sun_eci      Astronomical Almanac low-precision Sun (Vallado Alg. 29), ~0.01 deg
  gmst         IAU-82 GMST (Vallado eq. 3-45 form)
  shadow       conical umbra/penumbra (Montenbruck & Gill 3.4.2)
  b_field      IGRF-13 tilted dipole (g10, g11, h11 at 2025.0 incl. secular variation)
  density      exponential atmosphere at 550 km + diurnal bulge [WL]
"""
import numpy as np

MU = 3.986004418e14
RE = 6378137.0
J2 = 1.08262668e-3
RS = 6.957e8
AU = 1.495978707e11
P_SUN = 4.56e-6                       # [N/m^2] solar radiation pressure at 1 AU
JD_J2000 = 2451545.0
# IGRF-13 dipole coefficients at 2020.0 + 5 yr secular variation -> 2025.0 [nT]
G10, G11, H11 = -29404.8 + 5 * 5.7, -1450.9 + 5 * 7.4, 4652.5 + 5 * (-25.9)
M_DIP = np.array([G11, H11, G10])


def jd(t_unix):
    return np.asarray(t_unix) / 86400.0 + 2440587.5


def sun_eci(t_unix):
    """Unit Sun vector in ECI (mean equator of date ~ J2000) and distance [AU]."""
    n = jd(t_unix) - JD_J2000
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4.0e-7 * n)
    u = np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], -1)
    r_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    return u, r_au


def gmst(t_unix):
    return np.radians((280.46061837 + 360.98564736629 * (jd(t_unix) - JD_J2000)) % 360.0)


def rotz(a, v):
    """Rotate vector(s) v by angle(s) a about z (active rotation)."""
    a = np.asarray(a)
    c, s = np.cos(a), np.sin(a)
    v = np.asarray(v, float)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


class Orbit:
    """Circular orbit fixed in frame P (= Basilisk inertial frame) + J2 node rate for geometry."""

    def __init__(self, spec, t0_unix):
        self.t0 = t0_unix
        self.a = RE + 550e3
        self.n = np.sqrt(MU / self.a ** 3)
        if spec["kind"] == "sso":
            rate = 2 * np.pi / (365.2422 * 86400)
            cosi = -rate / (1.5 * self.n * J2 * (RE / self.a) ** 2)
            self.inc = np.arccos(cosi)
            # RAAN from local time of ascending node: RA(sun) + (LTAN - 12h) * 15 deg
            s, _ = sun_eci(t0_unix)
            ra_sun = np.arctan2(s[1], s[0])
            self.raan0 = ra_sun + np.radians((spec["ltan"] - 12.0) * 15.0)
        else:
            self.inc = np.radians(spec["inc"])
            self.raan0 = np.radians(spec["raan0"])
        self.argl0 = np.radians(spec.get("argl0", 0.0))
        self.raan_rate = -1.5 * self.n * J2 * (RE / self.a) ** 2 * np.cos(self.inc)
        self.period = 2 * np.pi / self.n

    def d_omega(self, t_unix):
        return self.raan_rate * (np.asarray(t_unix) - self.t0)

    def r_P(self, t_unix):
        u = self.argl0 + self.n * (np.asarray(t_unix) - self.t0)
        cO, sO, ci, si = np.cos(self.raan0), np.sin(self.raan0), np.cos(self.inc), np.sin(self.inc)
        cu, su = np.cos(u), np.sin(u)
        return self.a * np.stack([cO * cu - sO * su * ci, sO * cu + cO * su * ci, su * si], -1)

    def sun_P(self, t_unix):
        s, r_au = sun_eci(t_unix)
        return rotz(-self.d_omega(t_unix), s), r_au

    def eci_to_P(self, t_unix, v):
        return rotz(-self.d_omega(t_unix), v)

    def ecef_to_P(self, t_unix, v):
        return rotz(-(gmst(t_unix) + self.d_omega(t_unix)), v)

    def P_to_ecef(self, t_unix, v):
        return rotz(gmst(t_unix) + self.d_omega(t_unix), v)


def shadow(r, s_unit, r_au):
    """Fraction of the solar disc visible (1 sun, 0 umbra). r [m] (...,3), s_unit (...,3)."""
    r = np.asarray(r, float)
    d = s_unit * (r_au * AU)[..., None] - r
    rn, dn = np.linalg.norm(r, axis=-1), np.linalg.norm(d, axis=-1)
    a = np.arcsin(np.clip(RS / dn, -1, 1))
    b = np.arcsin(np.clip(RE / rn, -1, 1))
    c = np.arccos(np.clip(-(r * d).sum(-1) / (rn * dn), -1, 1))
    nu = np.ones_like(c)
    umb = c <= (b - a)
    part = (c < a + b) & ~umb
    if np.any(part):
        cp, ap, bp = c[part], a[part], b[part]
        x = (cp ** 2 + ap ** 2 - bp ** 2) / (2 * cp)
        y = np.sqrt(np.clip(ap ** 2 - x ** 2, 0, None))
        A = ap ** 2 * np.arccos(np.clip(x / ap, -1, 1)) + bp ** 2 * np.arccos(np.clip((cp - x) / bp, -1, 1)) - cp * y
        nu[part] = 1 - A / (np.pi * ap ** 2)
    nu[umb] = 0.0
    return nu


def b_field_ecef(r_ecef):
    """Tilted-dipole geomagnetic field [nT] at ECEF position(s) r [m]."""
    r = np.asarray(r_ecef, float)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    rh = r / rn
    mdotr = (rh * M_DIP).sum(-1, keepdims=True)
    return (RE / rn) ** 3 * (3 * mdotr * rh - M_DIP)


def b_field_P(orbit, t_unix, r_P):
    return orbit.ecef_to_P(t_unix, b_field_ecef(orbit.P_to_ecef(t_unix, r_P)))


def station_ecef(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return RE * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def elevation(orbit, t_unix, r_P, st_ecef):
    r_e = orbit.P_to_ecef(t_unix, r_P)
    d = r_e - st_ecef
    up = st_ecef / np.linalg.norm(st_ecef)
    return np.degrees(np.arcsin(np.clip((d * up).sum(-1) / np.linalg.norm(d, axis=-1), -1, 1)))


def density(r_P, s_P):
    """[kg/m^3] exponential atmosphere (rho(550 km) = 7e-13, H = 70 km) x diurnal bulge."""
    r = np.asarray(r_P, float)
    h = np.linalg.norm(r, axis=-1) - RE
    rho = 7e-13 * np.exp(-(h - 550e3) / 70e3)
    bulge = rotz(np.radians(30.0), s_P)              # bulge lags the Sun by ~2 h of local time
    cpsi = (r / np.linalg.norm(r, axis=-1, keepdims=True) * bulge).sum(-1)
    return rho * (1 + 0.35 * cpsi)


def mrp_to_dcm(s):
    """[BN] from MRP sigma_BN, vectorised (..., 3) -> (..., 3, 3) (Schaub & Junkins)."""
    s = np.asarray(s, float)
    s2 = (s * s).sum(-1)[..., None, None]
    x = np.zeros(s.shape[:-1] + (3, 3))
    x[..., 0, 1], x[..., 0, 2] = -s[..., 2], s[..., 1]
    x[..., 1, 0], x[..., 1, 2] = s[..., 2], -s[..., 0]
    x[..., 2, 0], x[..., 2, 1] = -s[..., 1], s[..., 0]
    I = np.eye(3)
    return I + (8 * x @ x - 4 * (1 - s2) * x) / (1 + s2) ** 2


def dcm_to_mrp(C):
    """MRP sigma for DCM [BN] (short set)."""
    C = np.asarray(C, float)
    tr = np.trace(C)
    zeta = np.sqrt(max(tr + 1.0, 1e-12))
    s = np.array([C[1, 2] - C[2, 1], C[2, 0] - C[0, 2], C[0, 1] - C[1, 0]]) / (zeta * (zeta + 2))
    if np.dot(s, s) > 1:
        s = -s / np.dot(s, s)
    return s
