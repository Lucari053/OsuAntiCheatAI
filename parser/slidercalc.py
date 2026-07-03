import math
from parser.curve import Bezier, Catmull
import bisect

# From https://github.com/Awlexus/python-osu-parser/blob/master/slidercalc.py
# Translated from JavaScript to Python by Awlex

class SliderPath:
    def __init__(self, curve_type, points):
        self.curve_type = curve_type
        self.points     = points
        self._poly      = None
        self._cum       = None

        if curve_type == "catmull":
            self._cache = Catmull(points)
        elif curve_type == "pass-through" and len(points) == 3:
            cc = get_circum_circle(*points)
            if cc is None:
                self.curve_type = "bezier"
                self._cache = self._build_bezier_segments(points)
                self._build_lut()
            else:
                self._cache = cc
        elif curve_type in ("bezier", "pass-through"):
            self._cache = self._build_bezier_segments(points)
            self._build_lut()
        else:
            self._cache = None # lienar
    
    def _build_lut(self):
        poly = []
        for seg in self._cache:
            pts = list(seg.pos.values())
            if poly and pts and poly[-1] == pts[0]:
                pts = pts[1:]
            poly.extend(pts)
        cum = [0.0]
        for i in range(1, len(poly)):
            cum.append(cum[-1] + math.hypot(poly[i][0] - poly[i-1][0],
                                            poly[i][1] - poly[i-1][1]))
        self._poly, self._cum = poly, cum

    @staticmethod
    def _build_bezier_segments(points):
        pts = points[:]
        segments = []
        previous = None
        i = 0
        while i < len(pts):
            point = pts[i]
            if previous is None:
                previous = point
                i += 1
                continue
            if point[0] == previous[0] and point[1] == previous[1]:
                segments.append(Bezier(pts[:i]))
                pts = pts[i:]
                i = 0
                previous = None
                continue
            previous = point
            i += 1
        segments.append(Bezier(pts))
        return segments
    
    def point_at_distance(self, distance):
        if self.curve_type == 'linear':
            return point_on_line(self.points[0], self.points[1], distance)

        if self.curve_type == 'catmull':
            return self._cache.point_at_distance(distance)

        if self.curve_type == 'pass-through' and len(self.points) == 3:
            cx, cy, radius = self._cache
            radians = distance / radius
            if is_left(*self.points):
                radians *= -1
            return rotate(cx, cy, self.points[0][0], self.points[0][1], radians)

        poly, cum = self._poly, self._cum
        if not poly:
            return None
        if distance <= 0:
            return list(poly[0])
        if distance >= cum[-1]:
            return list(poly[-1])
        j = bisect.bisect_left(cum, distance)
        seg = cum[j] - cum[j-1]
        u = (distance - cum[j-1]) / seg if seg > 0 else 0.0
        return [poly[j-1][0] + (poly[j][0] - poly[j-1][0]) * u,
                poly[j-1][1] + (poly[j][1] - poly[j-1][1]) * u]


def get_end_point(slider_type, slider_length, points):
    if not slider_type or not slider_length or not points:
        return
    return SliderPath(slider_type, points).point_at_distance(slider_length)


def point_on_line(p1, p2, length):
    full_length = math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))
    n = full_length - length

    x = (n * p1[0] + length * p2[0]) / full_length
    y = (n * p1[1] + length * p2[1]) / full_length
    return [x, y]


# Get coordinates of a point in a circle, given the center, a startpoint and a distance in radians
# @param {Float} cx       center x
# @param {Float} cy       center y
# @param {Float} x        startpoint x
# @param {Float} y        startpoint y
# @param {Float} radians  distance from the startpoint
# @return {Object} the new point coordinates after rotation
def rotate(cx, cy, x, y, radians):
    cos = math.cos(radians)
    sin = math.sin(radians)

    return [
        (cos * (x - cx)) - (sin * (y - cy)) + cx,
        (sin * (x - cx)) + (cos * (y - cy)) + cy
    ]


# Check if C is on left side of [AB]
# @param {Object} a startpoint of the segment
# @param {Object} b endpoint of the segment
# @param {Object} c the point we want to locate
# @return {Boolean} true if on left side
def is_left(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 0


# Get circum circle of 3 points
# @param  {Object} p1 first point
# @param  {Object} p2 second point
# @param  {Object} p3 third point
# @return {Object} circumCircle
def get_circum_circle(p1, p2, p3):
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    x3 = p3[0]
    y3 = p3[1]

    # center of circle
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6: # Colinear points -> not circle
        return None
    
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

    px = ux - x1
    py = uy - y1
    r = math.sqrt(px * px + py * py)

    return ux, uy, r
