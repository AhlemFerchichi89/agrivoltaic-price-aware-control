"""2-D periodic geometry of an elevated, south-facing, tilt-adjustable array.

The array is modelled as infinitely long east-west panel strips of slant width
w = GCR * pitch, pivoting about a horizontal axis at height H, repeated every
`pitch` metres in the north-south direction. Tilt theta > 0 turns the front
face to the south. All quantities are computed in the north-south vertical
cross-section (y = north, z = up), which is exact for infinitely long rows.

Outputs, for every hour and every candidate tilt:
  f_sh   fraction of the ground (crop) in the beam shadow of the panels
  svf    ground-averaged sky-view factor (diffuse light reaching the crop)
  cos_aoi, row_shade  incidence on the module and inter-row self-shading
"""
from __future__ import annotations

import numpy as np


class ArrayGeometry:
    def __init__(self, height: float, pitch: float, gcr: float, tilts_deg: np.ndarray, n_ground: int = 60):
        self.H = float(height)
        self.P = float(pitch)
        self.w = float(gcr) * float(pitch)
        self.tilts = np.asarray(tilts_deg, float)
        self.th = np.deg2rad(self.tilts)
        self.n_ground = n_ground
        self.svf = np.array([self._svf(t) for t in self.th])

    # ------------------------------------------------------------------ panel end points
    def _ends(self, th):
        c, s = np.cos(th), np.sin(th)
        y1, z1 = -self.w / 2 * c, self.H - self.w / 2 * s   # south (lower) edge
        y2, z2 = self.w / 2 * c, self.H + self.w / 2 * s    # north (upper) edge
        return y1, z1, y2, z2

    # ------------------------------------------------------------------ sky view factor
    def _svf(self, th: float) -> float:
        """Ground-averaged view factor to the sky (2-D, isotropic diffuse)."""
        y1, z1, y2, z2 = self._ends(th)
        ys = (np.arange(self.n_ground) + 0.5) / self.n_ground * self.P
        vals = []
        for y0 in ys:
            blocks = []
            for k in range(-40, 41):
                a1 = np.arctan2(y1 + k * self.P - y0, z1)
                a2 = np.arctan2(y2 + k * self.P - y0, z2)
                blocks.append((min(a1, a2), max(a1, a2)))
            blocks.sort()
            merged = []
            for a, b in blocks:
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            blocked = sum(0.5 * (np.sin(b) - np.sin(a)) for a, b in merged)
            vals.append(1 - blocked)
        return float(np.mean(vals))

    # ------------------------------------------------------------------ beam shading of the ground
    def ground_shade_fraction(self, sun_n: np.ndarray, sun_up: np.ndarray) -> np.ndarray:
        """Fraction of ground in shadow, shape (hours, tilts)."""
        up = np.maximum(sun_up, 1e-6)[:, None]
        k = (sun_n[:, None]) / up                    # shadow displacement per metre of height
        y1, z1, y2, z2 = self._ends(self.th[None, :])
        s1 = y1 - z1 * k
        s2 = y2 - z2 * k
        L = np.abs(s2 - s1)
        f = np.minimum(L, self.P) / self.P
        f[sun_up <= 0] = 0.0
        return f

    # ------------------------------------------------------------------ module incidence
    def cos_aoi(self, sun_n: np.ndarray, sun_up: np.ndarray) -> np.ndarray:
        """Cosine of incidence on the front face; normal = (y,z) = (-sin th, cos th)."""
        return np.clip(-sun_n[:, None] * np.sin(self.th)[None, :] + sun_up[:, None] * np.cos(self.th)[None, :], -1, 1)

    def row_shade_fraction(self, sun_n: np.ndarray, sun_up: np.ndarray) -> np.ndarray:
        """Fraction of a module shaded by the row to its south (beam only)."""
        th = self.th[None, :]
        c, s = np.cos(th), np.sin(th)
        up = np.maximum(sun_up, 1e-6)[:, None]
        n = sun_n[:, None]
        # top edge of the southern row
        yt, zt = -self.P + self.w / 2 * c, self.H + self.w / 2 * s
        # light travels along d = -(n, up); intersect p_top + t d with panel line (s_ c, H + s_ s)
        dy, dz = -n, -up
        # solve: yt + t dy = s_ c ; zt + t dz = H + s_ s
        det = dy * s - dz * c
        det = np.where(np.abs(det) < 1e-9, 1e-9, det)
        s_int = (dy * (zt - self.H) - dz * yt) / det
        frac = np.clip((s_int + self.w / 2) / self.w, 0, 1)
        frac = np.where((sun_n[:, None] < 0) & (sun_up[:, None] > 0), frac, 0.0)
        return frac

    def crossing_tilt(self, sun_n: np.ndarray, sun_up: np.ndarray) -> np.ndarray:
        """Tilt (deg) that is normal to the projected sun direction (max POA = max shade)."""
        prof = np.rad2deg(np.arctan2(sun_up, -sun_n))   # profile elevation measured from south
        return np.clip(90 - prof, self.tilts.min(), self.tilts.max())
