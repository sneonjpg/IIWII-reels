"""IIWII carousel 03: Tightrope Between Worlds. Clay / paper-craft miniature look, drawn in code.
Render at 2x (2160x2700), grain at final size, export 1080x1350 JPEG."""
import math, sys, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 2160, 2700
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
rng = np.random.default_rng(7)


def blur(a, r):
    r = max(0.1, r)
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), r)


def newmask():
    im = Image.new("L", (W, H), 0)
    return im, ImageDraw.Draw(im)


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


class Scene:
    def __init__(self):
        self.c = self.sky()

    def sky(self):
        # warm African dusk: indigo top -> plum -> rose -> amber glow low. No sun, moon or stars.
        stops = [(0.0, "#2f3157"), (0.28, "#5b4478"), (0.50, "#b0587a"), (0.68, "#e98a6a"), (0.82, "#f3b27a"), (1.0, "#f6c58f")]
        ys = np.linspace(0, 1, H)
        out = np.zeros((H, W, 3), np.float32)
        for i in range(len(stops) - 1):
            a, ca = stops[i]; b, cb = stops[i + 1]
            sel = (ys >= a) & (ys <= b)
            t = ((ys[sel] - a) / (b - a))[:, None]
            out[sel] = (hexc(ca) * (1 - t) + hexc(cb) * t)[:, None, :]
        # soft cloud bands
        im, d = newmask()
        for (x, y, w, h) in [(300, 1180, 900, 120), (1500, 1080, 1000, 140), (900, 2050, 1500, 150), (1700, 2250, 1100, 130), (200, 2300, 900, 120), (1000, 760, 800, 100), (1850, 640, 600, 90)]:
            d.ellipse((x - w / 2, y - h / 2, x + w / 2, y + h / 2), fill=150)
        m = blur(np.asarray(im, np.float32) / 255, 55)
        out = out * (1 - m[..., None] * 0.55) + hexc("#f2b9a0") * (m[..., None] * 0.55)
        return out

    def clay(self, mask, color, shadow=0.38, off=(16, 26), sh_blur=20, relief=1.0, shade_r=9, tint=None):
        m = np.asarray(mask, np.float32) / 255.0
        if shadow > 0:
            sm = blur(np.roll(np.roll(m, off[1], 0), off[0], 1), sh_blur)
            self.c = self.c * (1 - sm[..., None] * shadow) + hexc("#2a1a3a") * (sm[..., None] * shadow * 0.25)
        bm = blur(m, shade_r)
        gy, gx = np.gradient(bm)
        light = np.clip((-(gx * -0.7 + gy * -0.7)) * shade_r * 1.6, -1, 1) * relief  # light from top-left
        col = np.array(color, np.float32) if not isinstance(color, str) else hexc(color)
        shaded = col[None, None, :] * (1 + 0.42 * light[..., None])
        shaded = np.clip(shaded + 14 * np.clip(light, 0, 1)[..., None], 0, 255)
        a = blur(m, 1.2)[..., None]
        self.c = self.c * (1 - a) + shaded * a

    def glow(self, mask, color, strength=0.6, r=40):
        m = np.asarray(mask, np.float32) / 255.0
        g = blur(m, r)[..., None] * strength
        self.c = np.clip(self.c + hexc(color)[None, None, :] * g, 0, 255)

    def flat(self, mask, color, alpha=1.0):
        m = blur(np.asarray(mask, np.float32) / 255.0, 1.1)[..., None] * alpha
        col = np.array(color, np.float32) if not isinstance(color, str) else hexc(color)
        self.c = self.c * (1 - m) + col[None, None, :] * m


# ---------------------------------------------------------------- shapes
def rock(S, cx, cy, w, depth, seed, col="#8a5a44"):
    """floating clay island: flat top ellipse, tapered irregular underside."""
    r = random.Random(seed)
    im, d = newmask()
    pts = []
    n = 14
    for i in range(n + 1):
        t = i / n
        x = cx - w / 2 + w * t
        taper = math.sin(math.pi * t) ** 0.8
        y = cy + depth * taper * (0.75 + 0.25 * r.random())
        pts.append((x, y))
    top = [(cx - w / 2, cy), (cx + w / 2, cy)]
    d.polygon(pts + [(cx + w / 2, cy), (cx - w / 2, cy)], fill=255)
    d.ellipse((cx - w / 2, cy - w * 0.05, cx + w / 2, cy + w * 0.05), fill=255)
    S.clay(im, col, shadow=0.35, relief=1.2, shade_r=14)
    # darker lower band for depth
    im2, d2 = newmask()
    d2.polygon([(x, y) for x, y in pts] + [(cx + w / 2 * 0.7, cy + depth * 0.45), (cx - w / 2 * 0.7, cy + depth * 0.45)], fill=255)
    S.flat(im2, hexc("#4a2f3c"), 0.28)


def grass_top(S, cx, cy, w, col="#c9a66b"):
    im, d = newmask()
    d.ellipse((cx - w / 2 * 0.98, cy - w * 0.05, cx + w / 2 * 0.98, cy + w * 0.045), fill=255)
    S.clay(im, col, shadow=0.0, relief=0.8, shade_r=10)


def person(S, x, y, s=1.0, lean=0.0, arms=1.0):
    """tiny adolescent from behind, all-black hooded tracksuit, hood up, face never visible. (x,y)=feet on rope."""
    k = s
    tx = x + lean * 30 * k
    im, d = newmask()
    # legs (tracksuit trousers)
    d.rounded_rectangle((x - 30 * k, y - 125 * k, x - 3 * k, y), 10 * k, fill=255)
    d.rounded_rectangle((x + 3 * k, y - 125 * k, x + 30 * k, y), 10 * k, fill=255)
    # torso: wider at shoulders
    d.polygon([(x - 38 * k, y - 118 * k), (x + 38 * k, y - 118 * k), (tx + 50 * k, y - 205 * k), (tx - 50 * k, y - 205 * k)], fill=255)
    # arms out for balance, rounded ends
    for sgn in (-1, 1):
        sx, sy = tx + sgn * 46 * k, y - 195 * k
        ex, ey = tx + sgn * (128 + 20 * arms) * k, y - (150 - 14 * sgn * lean) * k
        d.line((sx, sy, ex, ey), fill=255, width=int(24 * k))
        d.ellipse((ex - 13 * k, ey - 13 * k, ex + 13 * k, ey + 13 * k), fill=255)
    # hood: big round hood over the head plus drape on the shoulders
    d.ellipse((tx - 42 * k, y - 292 * k, tx + 42 * k, y - 208 * k), fill=255)
    d.ellipse((tx - 62 * k, y - 232 * k, tx + 62 * k, y - 190 * k), fill=255)
    S.clay(im, "#2a2c38", shadow=0.4, off=(10, 16), sh_blur=9, relief=1.7, shade_r=5)
    # hood seam, subtle
    im2, d2 = newmask()
    d2.line((tx, y - 285 * k, tx, y - 205 * k), fill=255, width=max(2, int(3 * k)))
    S.flat(im2, hexc("#4a4d5e"), 0.8)


def sleeper(S, x, y, s=1.0):
    """asleep in hammock, hood over face, no skin. (x,y) = hammock centre."""
    k = s
    im, d = newmask()
    d.rounded_rectangle((x - 105 * k, y - 36 * k, x + 70 * k, y + 20 * k), 26 * k, fill=255)  # body
    d.ellipse((x + 40 * k, y - 54 * k, x + 115 * k, y + 14 * k), fill=255)  # hooded head
    d.rounded_rectangle((x - 135 * k, y - 16 * k, x - 95 * k, y + 26 * k), 12 * k, fill=255)  # feet
    S.clay(im, "#1a1b22", shadow=0.4, off=(8, 14), sh_blur=8, relief=0.9, shade_r=4)


def rope(S, p0, p1, ctrl, w=9):
    pts = []
    for i in range(0, 121):
        t = i / 120
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p1[1]
        pts.append((x, y))
    im, d = newmask()
    d.line(pts, fill=255, width=w, joint="curve")
    S.clay(im, "#d9b98a", shadow=0.3, off=(6, 16), sh_blur=10, relief=1.0, shade_r=3)
    # twisted fibre ticks
    im2, d2 = newmask()
    for i in range(0, 120, 2):
        (x, y) = pts[i]
        d2.line((x - 4, y - 5, x + 4, y + 5), fill=255, width=2)
    S.flat(im2, hexc("#8b6a43"), 0.5)
    return pts


def bez(p0, p1, ctrl, t):
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p1[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p1[1]
    return x, y


# ---------------------------------------------------------------- right oasis (always the same world)
def oasis(S, cx, cy, s=1.0, hammock_person=False):
    w = 1020 * s
    rock(S, cx, cy, w, 520 * s, 11, "#8a5a44")
    grass_top(S, cx, cy, w, "#efe0bf")  # white sand
    # lagoon
    im, d = newmask()
    d.ellipse((cx - 110 * s, cy - 18 * s, cx + 380 * s, cy + 48 * s), fill=255)
    S.clay(im, "#35c1c0", shadow=0.0, relief=-0.7, shade_r=6)
    im, d = newmask()
    d.ellipse((cx - 40 * s, cy - 6 * s, cx + 300 * s, cy + 26 * s), fill=255)
    S.flat(im, hexc("#8fe6dc"), 0.45)
    # palms
    palms = [(cx - 330 * s, cy - 6 * s, -0.12), (cx + 110 * s, cy - 8 * s, 0.14)]
    for (px, py, bend) in palms:
        im, d = newmask()
        pts = [(px + bend * 420 * s * (t ** 2), py - 430 * s * t) for t in np.linspace(0, 1, 18)]
        d.line(pts, fill=255, width=int(30 * s))
        S.clay(im, "#9c6a45", shadow=0.3, relief=1.1, shade_r=5)
        top = pts[-1]
        im, d = newmask()
        for ang in range(-170, 11, 30):
            a = math.radians(ang)
            ex, ey = top[0] + math.cos(a) * 190 * s, top[1] + math.sin(a) * 120 * s + 60 * s
            mx, my = top[0] + math.cos(a) * 95 * s, top[1] + math.sin(a) * 90 * s
            d.polygon([top, (mx - 34 * s * math.sin(a), my + 34 * s * math.cos(a)), (ex, ey), (mx + 34 * s * math.sin(a), my - 34 * s * math.cos(a))], fill=255)
        S.clay(im, "#3f9a63", shadow=0.28, relief=1.3, shade_r=6)
    # acacia silhouette (flat canopy)
    ax, ay = cx + 380 * s, cy - 4 * s
    im, d = newmask()
    d.polygon([(ax - 14 * s, ay), (ax + 14 * s, ay), (ax + 8 * s, ay - 250 * s), (ax - 18 * s, ay - 250 * s)], fill=255)
    d.line([(ax - 4 * s, ay - 120 * s), (ax - 90 * s, ay - 215 * s)], fill=255, width=int(12 * s))
    d.ellipse((ax - 170 * s, ay - 310 * s, ax + 150 * s, ay - 225 * s), fill=255)
    d.ellipse((ax - 90 * s, ay - 340 * s, ax + 90 * s, ay - 255 * s), fill=255)
    S.clay(im, "#4a3446", shadow=0.2, relief=1.1, shade_r=6)
    # woven hammock between palms
    hx0, hy0 = palms[0][0] + palms[0][2] * 420 * s * 0.45, cy - 215 * s
    hx1, hy1 = palms[1][0] + palms[1][2] * 420 * s * 0.45, cy - 215 * s
    pts = [bez((hx0, hy0), (hx1, hy1), ((hx0 + hx1) / 2, hy0 + 190 * s), t) for t in np.linspace(0, 1, 40)]
    im, d = newmask()
    d.line(pts, fill=255, width=int(26 * s))
    S.clay(im, "#d98a52", shadow=0.3, relief=1.0, shade_r=4)
    im, d = newmask()
    for i in range(0, 40, 2):
        x, y = pts[i]
        d.line((x - 8 * s, y - 10 * s, x + 8 * s, y + 10 * s), fill=255, width=3)
    S.flat(im, hexc("#fbe9c8"), 0.7)
    if hammock_person:
        mid = pts[20]
        sleeper(S, mid[0], mid[1] - 20 * s, 1.5 * s)
    return {"ham": pts[20], "left_edge": (cx - w / 2 + 40 * s, cy - 8 * s)}


# ---------------------------------------------------------------- left worlds
def laptop(S, x, y, w, tilt=0.0, glowc="#cfe9ff"):
    im, d = newmask()
    d.polygon([(x - w / 2, y), (x + w / 2, y), (x + w / 2 - 10, y + 22), (x - w / 2 + 10, y + 22)], fill=255)
    S.clay(im, "#3b3d4a", shadow=0.3, off=(8, 12), sh_blur=10, relief=1.2, shade_r=4)
    im, d = newmask()
    sw = w * 0.86
    d.polygon([(x - sw / 2, y - 5), (x + sw / 2, y - 5), (x + sw / 2 - 10 + tilt, y - w * 0.52), (x - sw / 2 + 10 + tilt, y - w * 0.52)], fill=255)
    S.clay(im, "#2b2d38", shadow=0.0, relief=1.0, shade_r=4)
    im, d = newmask()
    d.polygon([(x - sw / 2 + 16, y - 16), (x + sw / 2 - 16, y - 16), (x + sw / 2 - 24 + tilt, y - w * 0.46), (x - sw / 2 + 24 + tilt, y - w * 0.46)], fill=255)
    S.glow(im, glowc, 0.5, 42)
    S.flat(im, hexc(glowc), 1.0)


def laptop_tower(S, ox, oy, s, n=7):
    base = oy
    for i in range(n):
        w = (330 - i * 14) * s
        laptop(S, ox + (-1) ** i * (14 + 4 * i) * s, base - i * 108 * s, w, tilt=(-1) ** i * 6 * s, glowc=["#cfe9ff", "#ffe9b8", "#d8ffe9"][i % 3])


def calendar_city(S, ox, oy, s):
    # calendar page towers
    for (dx, h, w) in [(-190, 560, 210), (-10, 760, 230), (190, 420, 200)]:
        x = ox + dx * s
        im, d = newmask()
        d.rounded_rectangle((x - w / 2 * s, oy - h * s, x + w / 2 * s, oy), 12 * s, fill=255)
        S.clay(im, "#f6efe3", shadow=0.32, relief=1.0, shade_r=7)
        im, d = newmask()
        d.rounded_rectangle((x - w / 2 * s, oy - h * s, x + w / 2 * s, oy - (h - 70) * s), 12 * s, fill=255)
        S.clay(im, "#d9503f", shadow=0.0, relief=1.0, shade_r=4)
        im, d = newmask()
        for r_ in range(6):
            for c_ in range(4):
                gx = x - w / 2 * s + 24 * s + c_ * (w - 48) / 3 * s
                gy = oy - (h - 120) * s + r_ * 56 * s
                if gy < oy - 24 * s:
                    d.ellipse((gx - 7 * s, gy - 7 * s, gx + 7 * s, gy + 7 * s), fill=255)
        S.flat(im, hexc("#6b6477"), 0.7)
    # clock tower
    x = ox + 330 * s
    im, d = newmask()
    d.rounded_rectangle((x - 70 * s, oy - 700 * s, x + 70 * s, oy), 10 * s, fill=255)
    S.clay(im, "#cdbfa8", shadow=0.3, relief=1.0, shade_r=6)
    im, d = newmask()
    d.ellipse((x - 100 * s, oy - 830 * s, x + 100 * s, oy - 630 * s), fill=255)
    S.clay(im, "#f6efe3", shadow=0.3, relief=1.0, shade_r=6)
    im, d = newmask()
    cx0, cy0 = x, oy - 730 * s
    d.line((cx0, cy0, cx0, cy0 - 70 * s), fill=255, width=int(9 * s))
    d.line((cx0, cy0, cx0 + 52 * s, cy0 + 20 * s), fill=255, width=int(9 * s))
    S.flat(im, hexc("#2d2b3a"), 1.0)


def envelope(S, x, y, w, rot=0, col="#f3e7d1"):
    im, d = newmask()
    h = w * 0.64
    d.rounded_rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), 6, fill=255)
    S.clay(im, col, shadow=0.28, off=(8, 12), sh_blur=8, relief=1.0, shade_r=4)
    im, d = newmask()
    d.line([(x - w / 2 + 4, y - h / 2 + 4), (x, y + 2), (x + w / 2 - 4, y - h / 2 + 4)], fill=255, width=3)
    S.flat(im, hexc("#a5906f"), 0.8)


def phone(S, x, y, s):
    im, d = newmask()
    d.rounded_rectangle((x - 38 * s, y - 70 * s, x + 38 * s, y + 70 * s), 12 * s, fill=255)
    S.clay(im, "#23242d", shadow=0.3, off=(8, 12), sh_blur=8, relief=1.0, shade_r=4)
    im, d = newmask()
    d.rounded_rectangle((x - 31 * s, y - 60 * s, x + 31 * s, y + 60 * s), 6 * s, fill=255)
    S.glow(im, "#9fd2ff", 0.35, 26)
    S.flat(im, hexc("#9fd2ff"), 0.95)


def bubble(S, x, y, r, n="3"):
    im, d = newmask()
    d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    S.clay(im, "#e0393c", shadow=0.3, off=(6, 10), sh_blur=7, relief=1.3, shade_r=4)
    f = ImageFont.truetype(FONT, int(r * 1.15))
    im, d = newmask()
    d.text((x, y - r * 0.05), n, font=f, fill=255, anchor="mm")
    S.flat(im, hexc("#fff7ee"), 1.0)


def envelope_mountain(S, ox, oy, s):
    # triangle of stacked envelopes, phones and red bubbles
    r = random.Random(5)
    rows = 8
    for row in range(rows):
        n = rows - row
        for k in range(n):
            x = ox + (k - (n - 1) / 2) * 150 * s + r.uniform(-12, 12) * s
            y = oy - 40 * s - row * 92 * s
            envelope(S, x, y, 160 * s, col=["#f3e7d1", "#fff5e3", "#ead9bb"][(k + row) % 3])
    for (dx, dy) in [(-150, 250), (120, 470), (-40, 640)]:
        phone(S, ox + dx * s, oy - dy * s, 1.15 * s)
    for i, (dx, dy) in enumerate([(-250, 160), (200, 300), (0, 430), (-110, 560), (130, 650), (-30, 760)]):
        bubble(S, ox + dx * s, oy - dy * s, 34 * s, str(3 + i * 4))


def spreadsheet(S, x, y, w, h):
    im, d = newmask()
    d.rounded_rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), 5, fill=255)
    S.clay(im, "#f6f3ea", shadow=0.28, off=(6, 10), sh_blur=7, relief=0.9, shade_r=3)
    im, d = newmask()
    for i in range(1, 5):
        d.line((x - w / 2 + 4, y - h / 2 + i * h / 5, x + w / 2 - 4, y - h / 2 + i * h / 5), fill=255, width=2)
    for i in range(1, 4):
        d.line((x - w / 2 + i * w / 4, y - h / 2 + 4, x - w / 2 + i * w / 4, y + h / 2 - 4), fill=255, width=2)
    S.flat(im, hexc("#4a9a6a"), 0.8)


def sticky(S, x, y, w, col="#ffd24a"):
    im, d = newmask()
    d.rounded_rectangle((x - w / 2, y - w / 2, x + w / 2, y + w / 2), 4, fill=255)
    S.clay(im, col, shadow=0.28, off=(5, 9), sh_blur=6, relief=1.0, shade_r=3)


def tiny_tower(S, ox, oy, s):
    r = random.Random(2)
    for i in range(6):
        spreadsheet(S, ox + r.uniform(-10, 10) * s, oy - 30 * s - i * 62 * s, 200 * s, 52 * s)
    for (dx, dy, c) in [(-70, 60, "#ffd24a"), (60, 160, "#ff9eb5"), (-40, 290, "#9ee0a0")]:
        sticky(S, ox + dx * s, oy - dy * s, 42 * s, c)


def combined_tower(S, ox, oy, s):
    laptop_tower(S, ox, oy, s * 0.62, n=4)
    for k in range(3):
        for j in range(2 - k % 2 + 1):
            envelope(S, ox + (j - 1) * 120 * s + (k % 2) * 40 * s, oy - 520 * s - k * 70 * s, 130 * s, col=["#f3e7d1", "#fff5e3"][j % 2])
    sticky(S, ox - 140 * s, oy - 330 * s, 54 * s, "#ffd24a")
    sticky(S, ox + 150 * s, oy - 410 * s, 54 * s, "#ff9eb5")
    # calendar + clock on top
    im, d = newmask()
    d.rounded_rectangle((ox - 100 * s, oy - 880 * s, ox + 40 * s, oy - 720 * s), 10 * s, fill=255)
    S.clay(im, "#f6efe3", shadow=0.3, relief=1.0, shade_r=5)
    im, d = newmask()
    d.rounded_rectangle((ox - 100 * s, oy - 880 * s, ox + 40 * s, oy - 840 * s), 10 * s, fill=255)
    S.clay(im, "#d9503f", shadow=0.0, relief=1.0, shade_r=4)
    im, d = newmask()
    d.ellipse((ox + 40 * s, oy - 940 * s, ox + 200 * s, oy - 780 * s), fill=255)
    S.clay(im, "#f6efe3", shadow=0.3, relief=1.0, shade_r=5)
    im, d = newmask()
    cx0, cy0 = ox + 120 * s, oy - 860 * s
    d.line((cx0, cy0, cx0, cy0 - 50 * s), fill=255, width=int(7 * s))
    d.line((cx0, cy0, cx0 + 38 * s, cy0 + 12 * s), fill=255, width=int(7 * s))
    S.flat(im, hexc("#2d2b3a"), 1.0)


def clouds(S, spots):
    for (cx, cy, sc) in spots:
        im, d = newmask()
        for (dx, dy, r) in [(-120, 10, 60), (-50, -25, 80), (40, -15, 75), (120, 12, 58), (0, 20, 70)]:
            d.ellipse((cx + (dx - r) * sc, cy + (dy - r) * sc, cx + (dx + r) * sc, cy + (dy + r) * sc), fill=255)
        S.clay(im, "#f6d2c4", shadow=0.18, off=(10, 18), sh_blur=14, relief=1.0, shade_r=10)


# ---------------------------------------------------------------- headline
def headline(S, text, y0, size=170, maxw=1800):
    f = ImageFont.truetype(FONT, size)
    words, lines, cur = text.split(), [], ""
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    for w_ in words:
        t = (cur + " " + w_).strip()
        if probe.textlength(t, font=f) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w_
    lines.append(cur)
    im, d = newmask()
    y = y0
    for ln in lines:
        d.text((W / 2, y), ln, font=f, fill=255, anchor="mm")
        y += size * 1.12
    a = np.asarray(im, np.float32)
    a = blur(a, 5)
    a = (a > 110).astype(np.float32) * 255  # round corners: clay-soft lettering
    mask = Image.fromarray(blur(a, 3.0).clip(0, 255).astype(np.uint8))
    S.clay(mask, "#f6ecd6", shadow=0.5, off=(14, 30), sh_blur=22, relief=1.9, shade_r=12)


# ---------------------------------------------------------------- finish
def finish(S, path):
    c = S.c
    # calm, slightly dark base for the bottom 12 percent (logo zone), soft haze
    ys = np.linspace(0, 1, H)[:, None, None]
    fade = np.clip((ys - 0.82) / 0.18, 0, 1) ** 1.3
    c = c * (1 - fade * 0.28) + hexc("#3a2a4a") * (fade * 0.28)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W]
    v = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    c = c * (1 - 0.18 * np.clip(v - 0.4, 0, 1.2)[..., None])
    img = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).resize((1080, 1350), Image.LANCZOS)
    a = np.asarray(img, np.float32)
    g = rng.normal(0, 5.0, a.shape[:2])[..., None]  # film grain
    a = np.clip(a + g, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path, quality=93, subsampling=0)


# ---------------------------------------------------------------- slides
SL = {
    1: dict(text="Balance isn't 50/50.", left="laptops", L=(420, 1800, 0.92), R=(1660, 1820, 0.9), t=0.2, sag=190, shift=-150, lean=-0.5),
    2: dict(text="It's what today actually needs.", left="calendar", L=(420, 1800, 0.85), R=(1660, 1820, 0.9), t=0.5, sag=0, shift=0, lean=0.0),
    3: dict(text="Some days that's work.", left="mountain", L=(420, 1820, 0.95), R=(1880, 1700, 0.58), t=0.08, sag=50, shift=-120, lean=-0.8),
    4: dict(text="Some days that's rest.", left="tiny", L=(250, 1640, 0.5), R=(1520, 1840, 1.05), t=None, sag=70, shift=0, lean=0.0),
    5: dict(text="What does today need? Save this.", left="combined", L=(420, 1800, 0.92), R=(1660, 1820, 0.9), t=0.5, sag=0, shift=0, lean=0.0),
}


def build(n, out):
    cfg = SL[n]
    S = Scene()
    clouds(S, [(260, 1250, 1.3), (1850, 1120, 1.1), (1050, 1560, 0.9), (900, 2250, 1.0)] if n != 3 else [(260, 1250, 1.3), (1700, 1180, 1.0), (1050, 1560, 0.9)])
    lx, ly, ls = cfg["L"]
    rx, ry, rs = cfg["R"]
    # draw back to front: rocks, worlds, rope, figure, text
    rock(S, lx, ly, 760 * ls, 470 * ls, 3 + n, "#7e5a4c")
    grass_top(S, lx, ly, 760 * ls, "#b8a98f")
    {"laptops": laptop_tower, "calendar": calendar_city, "mountain": envelope_mountain, "tiny": tiny_tower, "combined": combined_tower}[cfg["left"]](S, lx, ly - 6 * ls, ls) if cfg["left"] != "laptops" else laptop_tower(S, lx, ly - 24 * ls, ls)
    info = oasis(S, rx, ry, rs, hammock_person=(n == 4))
    p0 = (lx + 340 * ls, ly - 14 * ls)
    p1 = info["left_edge"]
    mid = ((p0[0] + p1[0]) / 2 + cfg["shift"], (p0[1] + p1[1]) / 2 + cfg["sag"])
    rope(S, p0, p1, mid)
    if cfg["t"] is not None:
        fx, fy = bez(p0, p1, mid, cfg["t"])
        person(S, fx, fy, 1.05, lean=cfg["lean"])
    headline(S, cfg["text"], 400, size=200 if len(cfg["text"]) < 24 else 175)
    finish(S, out)



# ================================================================ v2: closer to the approved Canva test look
def _noise(sig, amp=1.0, seed=1):
    r = np.random.default_rng(seed)
    n = blur(r.normal(0, 1, (H, W)).astype(np.float32), sig)
    return n / (n.std() + 1e-6) * amp


_TEX = None


def new_sky(self):
    stops = [(0.0, "#46364a"), (0.22, "#6d4650"), (0.45, "#a9613f"), (0.68, "#d08a45"), (0.85, "#e0a455"), (1.0, "#e9b869")]
    ys = np.linspace(0, 1, H)
    out = np.zeros((H, W, 3), np.float32)
    for k in range(len(stops) - 1):
        a, ca = stops[k]; b, cb = stops[k + 1]
        sel = (ys >= a) & (ys <= b)
        t = ((ys[sel] - a) / (b - a))[:, None]
        out[sel] = (hexc(ca) * (1 - t) + hexc(cb) * t)[:, None, :]
    # soft atmospheric mottling so the sky is never flat
    out = out * (1 + 0.05 * _noise(120, 1, 3)[..., None]) 
    return out


def new_clay(self, mask, color, shadow=0.38, off=(16, 26), sh_blur=20, relief=1.0, shade_r=9, tint=None):
    global _TEX
    if _TEX is None:
        _TEX = _noise(2.2, 1, 5) * 0.6 + _noise(7, 1, 6) * 0.8
    m = np.asarray(mask, np.float32) / 255.0
    if shadow > 0:
        sm = blur(np.roll(np.roll(m, off[1], 0), off[0], 1), sh_blur)
        self.c = self.c * (1 - sm[..., None] * shadow) + hexc("#2a1a22") * (sm[..., None] * shadow * 0.25)
    bm = blur(m, shade_r)
    gy, gx = np.gradient(bm)
    light = np.clip((-(gx * -0.7 + gy * -0.7)) * shade_r * 1.6, -1, 1) * relief
    col = np.array(color, np.float32) if not isinstance(color, str) else hexc(color)
    shaded = col[None, None, :] * (1 + 0.42 * light[..., None] + 0.055 * _TEX[..., None])  # clay grain
    shaded = np.clip(shaded + 14 * np.clip(light, 0, 1)[..., None], 0, 255)
    a = blur(m, 1.2)[..., None]
    self.c = self.c * (1 - a) + shaded * a


Scene.sky = new_sky
Scene.clay = new_clay


def new_rock(S, cx, cy, w, depth, seed, col="#8a5a44"):
    r = random.Random(seed)
    im, d = newmask()
    n = 30
    pts = []
    for k in range(n + 1):
        t = k / n
        x = cx - w / 2 + w * t
        taper = math.sin(math.pi * t) ** 0.85
        y = cy + depth * taper * (0.62 + 0.38 * r.random())
        pts.append((x, y))
    d.polygon(pts + [(cx + w / 2, cy), (cx - w / 2, cy)], fill=255)
    d.ellipse((cx - w / 2, cy - w * 0.05, cx + w / 2, cy + w * 0.05), fill=255)
    S.clay(im, "#6a4a3e", shadow=0.4, relief=1.5, shade_r=12)
    # strata + crags
    im2, d2 = newmask()
    for k in range(5):
        yy = cy + depth * (0.16 + k * 0.15)
        d2.line([(cx - w * (0.44 - k * 0.07), yy + r.uniform(-14, 14)), (cx, yy + r.uniform(-24, 24)), (cx + w * (0.44 - k * 0.07), yy + r.uniform(-14, 14))], fill=255, width=7)
    S.flat(im2, hexc("#3a2630"), 0.35)


rock = new_rock


def new_clouds(S, spots, bank=True):
    def puff(cx, cy, sc, col="#d9b28a", sh=0.22):
        im, d = newmask()
        r = random.Random(int(cx + cy))
        for _ in range(9):
            dx, dy, rr = r.uniform(-150, 150), r.uniform(-45, 35), r.uniform(48, 92)
            d.ellipse((cx + (dx - rr) * sc, cy + (dy - rr) * sc, cx + (dx + rr) * sc, cy + (dy + rr) * sc), fill=255)
        S.clay(im, col, shadow=sh, off=(12, 20), sh_blur=16, relief=1.6, shade_r=11)
    if bank:  # low cloud sea, quiet so the bottom stays calm
        for k in range(9):
            puff(120 + k * 250, 2480 + (k % 3) * 60, 1.9, "#d9a66f", 0.1)
    for (cx, cy, sc) in spots:
        puff(cx, cy, sc)


clouds = new_clouds


def new_headline(S, text, y0, size=190, maxw=1900):
    f = ImageFont.truetype(FONT, size)
    lines = text.split("\n")
    im, d = newmask()
    y = y0
    for ln in lines:
        d.text((W / 2 + random.Random(len(ln)).uniform(-30, 30), y), ln, font=f, fill=255, anchor="mm")
        y += size * 1.1
    a = np.asarray(im, np.float32)
    a = (blur(a, 5) > 110).astype(np.float32) * 255
    # hand-sculpted wobble
    dx = _noise(60, 9, 11); dy = _noise(60, 9, 12)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = cv2.remap(a, xx + dx, yy + dy, cv2.INTER_LINEAR)
    mask = Image.fromarray(blur(a, 3.0).clip(0, 255).astype(np.uint8))
    S.clay(mask, "#ecd9b4", shadow=0.55, off=(14, 30), sh_blur=22, relief=2.0, shade_r=12)


headline = new_headline


def logo(img):
    """stand-in stacked IIWII lockup: swap for the real transparent PNG when supplied."""
    d = ImageDraw.Draw(img)
    cx, y = 540, 1228
    f = ImageFont.truetype("/usr/share/fonts/truetype/google-fonts/Poppins-Light.ttf", 52)
    tw = 104
    d.text((cx, y), "IIWII", font=f, fill=(255, 255, 255), anchor="mm")
    d.line((cx - tw / 2, y + 36, cx + tw / 2, y + 36), fill=(255, 255, 255), width=2)
    f2 = ImageFont.truetype("/usr/share/fonts/truetype/google-fonts/Poppins-Light.ttf", 9)
    d.text((cx, y + 49), "I T   I S   W H A T   I T   I S", font=f2, fill=(255, 255, 255), anchor="mm")


def new_finish(S, path):
    c = S.c
    ys = np.linspace(0, 1, H)[:, None, None]
    fade = np.clip((ys - 0.84) / 0.16, 0, 1) ** 1.3
    c = c * (1 - fade * 0.22) + hexc("#4a3040") * (fade * 0.22)
    c = c * np.array([1.03, 0.97, 0.90], np.float32)  # warm clay grade
    yy, xx = np.mgrid[0:H, 0:W]
    v = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    c = c * (1 - 0.2 * np.clip(v - 0.4, 0, 1.2)[..., None])
    img = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).resize((1080, 1350), Image.LANCZOS)
    a = np.asarray(img, np.float32)
    a = np.clip(a + rng.normal(0, 5.5, a.shape[:2])[..., None], 0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    logo(img)
    img.save(path, quality=93, subsampling=0)


finish = new_finish

TXT = {1: "Balance\nisn't\n50/50.", 2: "It's what\ntoday actually\nneeds.", 3: "Some days\nthat's work.", 4: "Some days\nthat's rest.", 5: "What does\ntoday need?\nSave this."}
for _k, _v in TXT.items():
    SL[_k]["text"] = _v
CL = {
    1: [(260, 540, 1.1), (1800, 480, 1.0), (1000, 1480, 0.8), (150, 1300, 1.1)],
    2: [(1850, 520, 1.1), (200, 560, 1.0), (1000, 1500, 0.8), (1900, 1250, 0.9)],
    3: [(1800, 470, 1.1), (250, 520, 1.0), (1150, 1350, 0.9)],
    4: [(1850, 470, 1.1), (250, 500, 1.1), (1000, 1480, 0.8)],
    5: [(1850, 500, 1.1), (220, 560, 1.0), (1000, 1480, 0.8)],
}


def build(n, out):
    cfg = SL[n]
    S = Scene()
    clouds(S, CL[n])
    lx, ly, ls = cfg["L"]
    rx, ry, rs = cfg["R"]
    rock(S, lx, ly, 760 * ls, 470 * ls, 3 + n, "#7e5a4c")
    grass_top(S, lx, ly, 760 * ls, "#b8a98f")
    if cfg["left"] == "laptops":
        laptop_tower(S, lx, ly - 24 * ls, ls)
    else:
        {"calendar": calendar_city, "mountain": envelope_mountain, "tiny": tiny_tower, "combined": combined_tower}[cfg["left"]](S, lx, ly - 6 * ls, ls)
    info = oasis(S, rx, ry, rs, hammock_person=(n == 4))
    p0 = (lx + 340 * ls, ly - 14 * ls)
    p1 = info["left_edge"]
    mid = ((p0[0] + p1[0]) / 2 + cfg["shift"], (p0[1] + p1[1]) / 2 + cfg["sag"])
    rope(S, p0, p1, mid)
    if cfg["t"] is not None:
        fx, fy = bez(p0, p1, mid, cfg["t"])
        person(S, fx, fy, 1.05, lean=cfg["lean"])
    headline(S, cfg["text"], 380, size=200)
    finish(S, out)



# ================================================================ v3 fixes
def new_clouds2(S, spots, bank=True):
    def puff(cx, cy, sc, col, sh, rel):
        im, d = newmask()
        r = random.Random(int(cx + cy))
        for _ in range(8):
            dx, dy, rr = r.uniform(-140, 140), r.uniform(-40, 30), r.uniform(55, 95)
            d.ellipse((cx + (dx - rr) * sc, cy + (dy - rr) * sc, cx + (dx + rr) * sc, cy + (dy + rr) * sc), fill=255)
        S.clay(im, col, shadow=sh, off=(12, 20), sh_blur=18, relief=rel, shade_r=14)
    if bank:  # calm low cloud sea behind the logo zone
        for k in range(8):
            puff(60 + k * 290, 2690 + (k % 2) * 30, 2.2, "#dcab6a", 0.06, 0.6)
    for (cx, cy, sc) in spots:
        puff(cx, cy, sc, "#dcb68e", 0.2, 1.0)


clouds = new_clouds2


def combined_tower(S, ox, oy, s):
    envelope_mountain(S, ox, oy, s * 0.55)
    top = oy - (40 + 7 * 92) * 0.55 * s
    laptop_tower(S, ox, top - 6 * s, s * 0.6, n=3)
    ct = top - 3 * 108 * 0.6 * s - 30 * s
    im, d = newmask()
    d.rounded_rectangle((ox - 150 * s, ct - 170 * s, ox + 10 * s, ct), 10 * s, fill=255)
    S.clay(im, "#f6efe3", shadow=0.3, relief=1.0, shade_r=5)
    im, d = newmask()
    d.rounded_rectangle((ox - 150 * s, ct - 170 * s, ox + 10 * s, ct - 125 * s), 10 * s, fill=255)
    S.clay(im, "#d9503f", shadow=0.0, relief=1.0, shade_r=4)
    im, d = newmask()
    d.ellipse((ox + 10 * s, ct - 250 * s, ox + 190 * s, ct - 70 * s), fill=255)
    S.clay(im, "#f6efe3", shadow=0.3, relief=1.0, shade_r=5)
    im, d = newmask()
    cx0, cy0 = ox + 100 * s, ct - 160 * s
    d.line((cx0, cy0, cx0, cy0 - 56 * s), fill=255, width=int(8 * s))
    d.line((cx0, cy0, cx0 + 42 * s, cy0 + 14 * s), fill=255, width=int(8 * s))
    S.flat(im, hexc("#2d2b3a"), 1.0)
    sticky(S, ox - 215 * s, oy - 230 * s, 64 * s, "#ffd24a")
    sticky(S, ox + 235 * s, oy - 180 * s, 64 * s, "#ff9eb5")
    sticky(S, ox - 190 * s, oy - 330 * s, 56 * s, "#9ee0a0")


SL[1].update(L=(400, 1750, 1.05), R=(1640, 1770, 1.05))
SL[2].update(L=(400, 1750, 0.95), R=(1640, 1770, 1.05))
SL[3].update(L=(400, 1770, 1.05), R=(1880, 1650, 0.66))
SL[4].update(L=(240, 1600, 0.55), R=(1500, 1780, 1.12))
SL[5].update(L=(400, 1750, 1.05), R=(1640, 1770, 1.05))
CL = {
    1: [(1850, 760, 0.9), (200, 880, 0.9), (1100, 1450, 0.7)],
    2: [(1900, 800, 0.9), (180, 900, 0.9), (1100, 1450, 0.7)],
    3: [(1850, 820, 0.9), (150, 900, 0.8), (1150, 1300, 0.8)],
    4: [(1900, 800, 0.9), (200, 900, 0.9), (1000, 1400, 0.7)],
    5: [(1900, 800, 0.9), (180, 900, 0.9), (1100, 1450, 0.7)],
}


if __name__ == "__main__":
    n = int(sys.argv[1]); out = sys.argv[2]
    build(n, out)
    print("ok", out)
