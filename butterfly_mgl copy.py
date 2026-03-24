# -*- coding: utf-8 -*-
# butterfly_swarm_4d.py — рой 4D‑бабочек со шлейфами и «шипастыми» звёздами
# + процедурная туманность (светло‑серая), HUD со здоровьем и управление «игроком».
# pip install vispy numpy PySide6   (или PyQt5)

import os, time
import numpy as np

# ---- выбрать Qt-бэкенд ----
try:
    import PySide6
    from vispy import app
    app.use_app('pyside6')
    pyside_dir = os.path.dirname(PySide6.__file__)
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                          os.path.join(pyside_dir, "plugins", "platforms"))
except Exception:
    try:
        import PyQt5
        from vispy import app
        app.use_app('pyqt5')
    except Exception:
        from vispy import app

os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from vispy import scene, color
from vispy.scene import visuals
from vispy.visuals import transforms as tr

# -------------------- математика/геометрия --------------------
def rotate_plane(P, i, j, theta):
    c, s = np.cos(theta), np.sin(theta)
    Pi = P[:, i].copy()
    Pj = P[:, j].copy()
    P[:, i] = c * Pi - s * Pj
    P[:, j] = s * Pi + c * Pj
    return P

def project_4d_to_3d(P4, d=3.5):
    w = P4[:, 3]
    denom = (d - w)
    denom[np.abs(denom) < 1e-6] = 1e-6
    factor = (d / denom).reshape(-1, 1)
    return P4[:, :3] * factor

def project_point4_to_3(c4, d=3.5):
    w = c4[3]
    denom = (d - w) if abs(d - w) > 1e-6 else 1e-6
    return c4[:3] * (d / denom)

def butterfly_4d(n=1500, scale=0.20):
    t = np.linspace(0, 24*np.pi, n).astype(np.float32)
    r = np.exp(np.cos(t)) - 2*np.cos(4*t) + np.sin(t/12)**5
    x = r*np.sin(t); y = r*np.cos(t)
    z = 0.55*np.sin(3*t); w = 0.55*np.cos(2*t)
    P4 = np.stack([x, y, z, w], axis=1).astype(np.float32)
    P4 /= (np.max(np.linalg.norm(P4[:, :3], axis=1)) + 1e-6)
    P4 *= scale
    return P4, t

# -------------------- параметры --------------------
MAX_BUTTERFLIES      = 18
INIT_BUTTERFLIES     = 5
POINTS_PER_BUTTERFLY = 1500

# звёзды
STAR_MAX        = 30
STAR_SPAWN_RATE = 0.55
STAR_TTL        = (7.0, 13.0)
STAR_RADIUS     = 0.20

# «шипастые» звезды
STAR_SPIKES_MIN = 12
STAR_SPIKES_MAX = 20
SPIKE_INNER_FR  = 0.35
SPIKES_WIDTH_BASE = 1.8   # px при высоте 900
STAR_CORE_SIZE   = 12.0   # px при высоте 900

# жизнь/размножение
SPLIT_BOOST     = 0.40
SPLIT_PROB      = 0.60
REPRO_COOLDOWN  = (2.0, 4.0)
FADE_RATE       = 0.030
BASE_FADE       = 0.010
COLLISION_RADIUS     = 0.22
COLLISION_BIRTH_PROB = 0.015
VOID_BIRTH_RATE      = 0.012

# динамика
OU_BETA       = 0.9
OU_SIGMA      = 0.6
SPEED_LIMIT   = 0.70
ACCEL_LIMIT   = 1.0
SEEK_GAIN     = 1.1

# управление игроком
PLAYER_IMPULSE = 2.0
PLAYER_BOOST   = 2.0  # с шифтом

# wander
WANDER_GAIN   = 0.35
WANDER_T_MIN  = 3.0
WANDER_T_MAX  = 6.0

# «бабочья» физика
FLAP_FREQ     = (1.4, 2.6)
FLAP_FORCE    = 0.75
LIFT_GAIN     = 0.30
GLIDE_DRAG    = 0.22

# деформация формы
WING_AMP      = 0.28
W_OSC         = 1.0

# визуал/свечение
BASE_MARKER_SIZE = 2.1
BASE_TRAIL_WIDTH = 1.8
GLOW_MARKER_SCALE = 2.4
GLOW_LINE_SCALE   = 2.6
HALO_ALPHA        = 0.18
MIN_ALPHA         = 0.25
BG_COLOR = 'black'

# Туманность — светло‑серая, прозрачная
NEBULA_POINTS      = 2400
NEBULA_STRENGTH    = 0.20   # 1/2 — меньше/больше
NEBULA_RADIUS_MIN  = 7.0
NEBULA_RADIUS_MAX  = 14.0
NEBULA_COLOR_A     = np.array([0.58, 0.60, 0.64], np.float32)
NEBULA_COLOR_B     = np.array([0.34, 0.36, 0.40], np.float32)
NEBULA_GLOW_SCALE  = 2.4
NEBULA_BASE_ALPHA  = 0.42   # базовая прозрачность (потом умножается на шум/дальность)

BASE_SHAPE, _t = butterfly_4d(POINTS_PER_BUTTERFLY, scale=0.20)
_rng = np.random.default_rng()

# -------------------- сцена --------------------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True, bgcolor=BG_COLOR)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=55, distance=7.5, elevation=22, azimuth=35)

# ----- TUMAN (NEBULA) -----
class Nebula:
    def __init__(self, parent):
        u = _rng.uniform(-1, 1, NEBULA_POINTS)
        theta = _rng.uniform(0, 2*np.pi, NEBULA_POINTS)
        phi = np.arccos(u).astype(np.float32)
        r = _rng.uniform(NEBULA_RADIUS_MIN, NEBULA_RADIUS_MAX, NEBULA_POINTS).astype(np.float32)
        pos = np.zeros((NEBULA_POINTS, 3), np.float32)
        pos[:,0] = r*np.sin(phi)*np.cos(theta)
        pos[:,1] = r*np.sin(phi)*np.sin(theta)
        pos[:,2] = r*np.cos(phi)
        self.pos = pos
        base_size = (np.interp(r, [NEBULA_RADIUS_MIN, NEBULA_RADIUS_MAX], [36.0, 72.0])).astype(np.float32)
        self.size_core = base_size
        self.size_halo = base_size * NEBULA_GLOW_SCALE
        mix_k = _rng.uniform(0.0, 1.0, (NEBULA_POINTS,1)).astype(np.float32)
        rgb = NEBULA_COLOR_A*(1.0-mix_k) + NEBULA_COLOR_B*mix_k
        self.color_rgb = rgb
        self.halo = scene.visuals.Markers(parent=parent); self.halo.antialias = 1
        self.core = scene.visuals.Markers(parent=parent); self.core.antialias = 1
        self.halo.set_gl_state(blend=True, depth_test=True)
        self.core.set_gl_state(blend=True, depth_test=True)
        self.j1 = _rng.uniform(0, 1000, (NEBULA_POINTS, 3)).astype(np.float32)
        self.j2 = _rng.uniform(0, 1000, (NEBULA_POINTS, 3)).astype(np.float32)

    @staticmethod
    def _fbm_like(p, t):
        f1 = np.cos((p[:,0]*0.15 + p[:,1]*0.11 + p[:,2]*0.09) + t*0.20)*0.5 + 0.5
        f2 = np.cos((p[:,0]*0.32 + p[:,1]*0.22 - p[:,2]*0.18) - t*0.12)*0.5 + 0.5
        f3 = np.cos((p[:,0]*0.07 - p[:,1]*0.27 + p[:,2]*0.19) + t*0.31)*0.5 + 0.5
        v = (0.55*f1 + 0.30*f2 + 0.15*f3)
        return v.astype(np.float32)

    def update(self, t, cam_pos, strength, size_scale):
        p = self.pos
        fbm = self._fbm_like(p + self.j1, t) * self._fbm_like(p*0.65 + self.j2, t*0.7)
        dist = np.linalg.norm(p - cam_pos[None, :], axis=1)
        dist_k = np.clip((dist - 4.0) / 8.0, 0.25, 1.0)
        alpha = NEBULA_BASE_ALPHA * fbm * dist_k * strength
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        core_col = np.concatenate([self.color_rgb, alpha[:,None]], 1)
        halo_col = core_col.copy(); halo_col[:,3] *= 0.35
        self.halo.set_data(pos=p, face_color=halo_col,
                           size=(self.size_halo*size_scale).astype(np.float32), edge_width=0.0)
        self.core.set_data(pos=p, face_color=core_col,
                           size=(self.size_core*size_scale).astype(np.float32), edge_width=0.0)

nebula = Nebula(view.scene)

# ----- Бабочки/звезды -----
butter_viz = scene.visuals.Markers(parent=view.scene); butter_viz.antialias = 1
butter_halo_viz = scene.visuals.Markers(parent=view.scene); butter_halo_viz.antialias = 1
stars_core_viz = scene.visuals.Markers(parent=view.scene); stars_core_viz.antialias = 1
stars_halo_viz = scene.visuals.Markers(parent=view.scene); stars_halo_viz.antialias = 1
stars_spikes_viz = visuals.Line(parent=view.scene, method='gl')
stars_spikes_viz.set_gl_state(blend=True, depth_test=True)

# ----- HUD (оverlay в экранных координатах) -----
hud_root = scene.Node(parent=canvas.scene)
hud_root.transform = tr.STTransform()
hud_text = visuals.Text("", color='white', parent=hud_root, anchor_x='left', anchor_y='top', font_size=12)
hud_text.pos = (12, 22)
# рамка и заполнение полоски здоровья (в пикселях)
hud_frame = visuals.Rectangle(center=(120, 42), width=180, height=12,
                              radius=4, color=None, border_color=(0.8,0.8,0.85,0.9),
                              parent=hud_root)
hud_fill  = visuals.Rectangle(center=(120, 42), width=1, height=8,
                              radius=3, color=(0.25,0.85,0.55,0.95),
                              parent=hud_root)
def _hud_on_resize():
    w, h = canvas.size
    hud_root.transform.scale = (1, 1, 1)
    hud_root.transform.translate = (0, h, 0)  # anchor сверху‑слева
    # текст и бар уже в пиксельных координатах

@canvas.events.resize.connect
def _on_canvas_resize(ev):
    _apply_scale_from_canvas()
    _hud_on_resize()

# список бабочек
butterflies = []

# динамическое масштабирование
_current_marker_size = 1.0
_current_trail_w     = 1.0
_current_star_base   = 1.0
_current_spikes_w    = 1.0
_current_nebula_scale= 1.0

def _apply_scale_from_canvas():
    global _current_marker_size, _current_trail_w, _current_star_base, _current_spikes_w, _current_nebula_scale
    _, h = canvas.size
    scale = max(0.33, min(3.0, h/900.0))
    _current_marker_size = BASE_MARKER_SIZE * scale
    _current_trail_w     = BASE_TRAIL_WIDTH * scale
    _current_star_base   = STAR_CORE_SIZE * scale
    _current_spikes_w    = SPIKES_WIDTH_BASE * scale
    _current_nebula_scale= scale

_apply_scale_from_canvas(); _hud_on_resize()

# -------------------- сущности --------------------
class Star:
    __slots__ = ("pos3","ttl","born","alive","hue_jitter","dirs")
    def __init__(self):
        r = _rng.uniform(0.3, 1.0)
        theta = _rng.uniform(0, 2*np.pi)
        phi = np.arccos(_rng.uniform(-1, 1))
        radius = 3.2 * r**1.6
        x = radius*np.sin(phi)*np.cos(theta); y = radius*np.sin(phi)*np.sin(theta); z = radius*np.cos(phi)
        self.pos3 = np.array([x, y, z], np.float32)
        self.ttl  = float(_rng.uniform(*STAR_TTL))
        self.born = time.perf_counter()
        self.alive = True
        self.hue_jitter = float(_rng.uniform(-0.03, 0.03))
        k = _rng.integers(STAR_SPIKES_MIN, STAR_SPIKES_MAX+1)
        u = _rng.normal(0, 1, size=(k, 3)).astype(np.float32)
        u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-6)
        self.dirs = u
    def update(self, dt):
        self.ttl -= dt
        if self.ttl <= 0: self.alive = False

class Butterfly:
    __slots__ = ("center4","vel4","acc4","angle1","angle2","vital","fade_in",
                 "target_id","scale","mode","mode_ttl","wander_center",
                 "wander_omega","repro_cd","palette","hue0","sat","val",
                 "flap_phase","flap_omega","trail3","trail_idx","trail_timer",
                 "line","halo")
    def __init__(self, center4=None, copy_from=None):
        if copy_from is not None:
            self.center4 = copy_from.center4.copy()
            self.vel4    = copy_from.vel4.copy()*0.0
            self.acc4    = np.zeros(4, np.float32)
            self.angle1  = copy_from.angle1 + float(_rng.uniform(-0.25, 0.25))
            self.angle2  = copy_from.angle2 + float(_rng.uniform(-0.25, 0.25))
            self.vital   = min(1.0, copy_from.vital*0.7 + 0.25)
            self.fade_in = 0.0
            self.scale   = copy_from.scale * float(_rng.uniform(0.95, 1.05))
        else:
            self.center4 = (_rng.uniform(-1.0, 1.0, 4).astype(np.float32)
                            if center4 is None else center4.astype(np.float32))
            self.vel4    = _rng.normal(0, 0.25, 4).astype(np.float32)
            self.acc4    = np.zeros(4, np.float32)
            self.angle1  = float(_rng.uniform(0, 2*np.pi))
            self.angle2  = float(_rng.uniform(0, 2*np.pi))
            self.vital   = 0.40
            self.fade_in = 0.0
            self.scale   = np.float32(1.0)

        self.target_id = -1
        self.mode      = 'seek' if _rng.random() < 0.6 else 'wander'
        self.mode_ttl  = float(_rng.uniform(WANDER_T_MIN, WANDER_T_MAX))
        self.wander_center = self.center4[:3].copy()
        self.wander_omega  = _rng.uniform(0.15, 0.35, 3).astype(np.float32)
        self.repro_cd = float(_rng.uniform(*REPRO_COOLDOWN))

        if _rng.random() < 0.5:
            self.palette = None
            self.hue0 = float(_rng.uniform(0, 1.0))
            self.sat  = float(_rng.uniform(0.65, 0.95))
            self.val  = float(_rng.uniform(0.80, 0.98))
        else:
            cmaps = ['viridis','plasma','magma','inferno','turbo','twilight','ice','winter','autumn','spring']
            self.palette = color.get_colormap(_rng.choice(cmaps))
            self.hue0 = self.sat = self.val = 0.0

        self.flap_phase = float(_rng.uniform(0, 2*np.pi))
        self.flap_omega = 2*np.pi*float(_rng.uniform(*FLAP_FREQ))

        self.trail3 = np.zeros((64, 3), np.float32)
        self.trail_idx = 0
        self.trail_timer = 0.0
        self.line = visuals.Line(parent=view.scene, method='gl', width=BASE_TRAIL_WIDTH)
        self.halo = visuals.Line(parent=view.scene, method='gl', width=BASE_TRAIL_WIDTH*GLOW_LINE_SCALE)
        self.line.set_gl_state(blend=True, depth_test=True)
        self.halo.set_gl_state(blend=True, depth_test=True)

    def pick_target(self, stars):
        if not stars:
            self.target_id = -1; return
        C3 = self.center4[:3]
        d = [np.linalg.norm(C3 - s.pos3) for s in stars]
        self.target_id = int(np.argmin(d))

    def _wander_point(self, t):
        base = self.wander_center
        osc  = np.sin(t*self.wander_omega + np.array([0.0, 2.1, 4.2])) * 0.6
        return base + osc.astype(np.float32)

    def step(self, dt, stars, player_impulse=None):
        tnow = time.perf_counter()
        self.center4[3] += 0.18*np.sin(W_OSC*tnow)*dt

        self.mode_ttl -= dt
        if self.mode_ttl <= 0:
            self.mode = 'wander' if self.mode == 'seek' else 'seek'
            self.mode_ttl = float(_rng.uniform(WANDER_T_MIN, WANDER_T_MAX))
            if self.mode == 'wander':
                self.wander_center = self.center4[:3] + _rng.normal(0, 0.6, 3).astype(np.float32)

        if self.mode == 'seek':
            if self.target_id < 0 or self.target_id >= len(stars) or not stars[self.target_id].alive:
                self.pick_target(stars)

        self.acc4 *= 0.0
        self.acc4 += _rng.normal(0.0, OU_SIGMA, 4).astype(np.float32) * np.sqrt(max(dt,1e-3))
        self.acc4 += -OU_BETA*self.vel4

        self.flap_phase += self.flap_omega * dt
        flap = 0.5*(1.0 - np.cos(self.flap_phase))
        v3 = self.vel4[:3]; vnorm = np.linalg.norm(v3) + 1e-6
        heading = v3 / vnorm
        self.acc4[:3] += FLAP_FORCE * flap * heading
        self.acc4[2]  += LIFT_GAIN  * flap

        if self.mode == 'seek' and 0 <= self.target_id < len(stars) and stars[self.target_id].alive:
            dir3 = (stars[self.target_id].pos3 - self.center4[:3])
            self.acc4[:3] += SEEK_GAIN * dir3
        else:
            want = self._wander_point(tnow) - self.center4[:3]
            self.acc4[:3] += WANDER_GAIN * want

        # импульс игрока
        if player_impulse is not None:
            self.acc4[:3] += player_impulse

        an = np.linalg.norm(self.acc4)
        if an > ACCEL_LIMIT: self.acc4 *= ACCEL_LIMIT/(an+1e-6)
        self.vel4 += self.acc4 * dt
        if self.mode != 'seek': self.vel4 *= (1.0 - GLIDE_DRAG*dt)
        vn = np.linalg.norm(self.vel4)
        if vn > SPEED_LIMIT: self.vel4 *= (SPEED_LIMIT/(vn+1e-6))
        self.center4 += self.vel4 * dt

        self.angle1 += 0.8*dt
        self.angle2 += 0.55*dt

        self.vital = max(0.0, self.vital - BASE_FADE*dt - (FADE_RATE*dt if self.mode=='seek' and self.target_id<0 else 0.0))
        self.fade_in = min(1.0, self.fade_in + 0.8*dt)

        self.trail_timer += dt
        if self.trail_timer >= 0.03:
            self.trail_timer = 0.0
            c3 = project_point4_to_3(self.center4)
            self.trail3[self.trail_idx] = c3
            self.trail_idx = (self.trail_idx + 1) % len(self.trail3)

        if self.repro_cd > 0: self.repro_cd -= dt

    def try_consume_star(self, stars):
        if not (0 <= self.target_id < len(stars)): return False, False
        st = stars[self.target_id]
        if not st.alive: return False, False
        if np.linalg.norm(self.center4[:3] - st.pos3) <= STAR_RADIUS:
            st.alive = False
            self.vital = min(1.0, self.vital + SPLIT_BOOST)
            spawn = (self.repro_cd <= 0) and (_rng.random() < SPLIT_PROB)
            if spawn: self.repro_cd = float(_rng.uniform(*REPRO_COOLDOWN))
            return True, spawn
        return False, False

    def alive(self): return self.vital > 0.02

class StarManager:
    def __init__(self):
        self.items = [Star() for _ in range(10)]
        self.spawn_acc = 0.0
    def update(self, dt):
        self.spawn_acc += STAR_SPAWN_RATE * dt
        while self.spawn_acc >= 1.0 and len(self.items) < STAR_MAX:
            self.items.append(Star()); self.spawn_acc -= 1.0
        for s in self.items: s.update(dt)
        self.items = [s for s in self.items if s.alive]

stars = StarManager()
last_time = time.perf_counter()
void_birth_acc = 0.0

def spawn_butterfly_near(parent):
    b = Butterfly(copy_from=parent)
    b.center4[:3] += _rng.normal(0, 0.05, 3).astype(np.float32)
    b.center4[3]  += _rng.normal(0, 0.03)
    return b

def spawn_butterfly_void():
    c4 = _rng.uniform(-1.2, 1.2, 4).astype(np.float32)
    b = Butterfly(center4=c4)
    b.vital = 0.25
    b.fade_in = 0.0
    return b

def hsv_to_rgb(h, s, v):
    i = np.floor(h*6).astype(np.int32); f = h*6 - i
    p = v*(1-s); q = v*(1-f*s); t = v*(1-(1-f)*s); i%=6
    out = np.zeros((h.shape[0],3),np.float32)
    m=(i==0); out[m]=np.stack([v[m],t[m],p[m]],1)
    m=(i==1); out[m]=np.stack([q[m],v[m],p[m]],1)
    m=(i==2); out[m]=np.stack([p[m],v[m],t[m]],1)
    m=(i==3); out[m]=np.stack([p[m],q[m],v[m]],1)
    m=(i==4); out[m]=np.stack([t[m],p[m],v[m]],1)
    m=(i==5); out[m]=np.stack([v[m],p[m],q[m]],1)
    return out

# стартовый рой
butterflies = [spawn_butterfly_void() for _ in range(INIT_BUTTERFLIES)]

# --- управление/хоткеи ---
keys_down = set()
player_idx = 0  # индекс «игрока» (TAB меняет), None — без игрока
hud_on = True

def on_key_press(ev):
    global NEBULA_STRENGTH, player_idx, hud_on
    k = ev.key.upper() if isinstance(ev.key, str) else ev.key
    keys_down.add(k)
    if k == 'TAB':
        if butterflies:
            player_idx = 0 if player_idx is None else (player_idx + 1) % len(butterflies)
    elif k == 'H':
        hud_on = not hud_on
        # спрячем HUD
        for v in (hud_text, hud_frame, hud_fill):
            v.visible = hud_on
    elif k == '1':
        NEBULA_STRENGTH = float(np.clip(NEBULA_STRENGTH - 0.03, 0.0, 0.6))
        print('NEBULA_STRENGTH =', NEBULA_STRENGTH)
    elif k == '2':
        NEBULA_STRENGTH = float(np.clip(NEBULA_STRENGTH + 0.03, 0.0, 0.6))
        print('NEBULA_STRENGTH =', NEBULA_STRENGTH)

def on_key_release(ev):
    k = ev.key.upper() if isinstance(ev.key, str) else ev.key
    if k in keys_down: keys_down.remove(k)

canvas.events.key_press.connect(on_key_press)
canvas.events.key_release.connect(on_key_release)

def _player_impulse_vec():
    if not butterflies or player_idx is None: return None
    dx = (-1 if 'A' in keys_down else 0) + (1 if 'D' in keys_down else 0)
    dy = (-1 if 'Q' in keys_down else 0) + (1 if 'E' in keys_down else 0)
    dz = (-1 if 'W' in keys_down else 0) + (1 if 'S' in keys_down else 0)  # вперёд W: -Z
    v = np.array([dx, dy, dz], np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6: return None
    v /= n
    boost = PLAYER_BOOST if ('SHIFT' in keys_down or 'SHIFT_R' in keys_down) else 1.0
    return v * (PLAYER_IMPULSE * boost)

def update(ev):
    global last_time, void_birth_acc, butterflies, player_idx
    t = time.perf_counter()
    dt = min(0.05, max(1e-4, t - last_time))
    last_time = t

    cam = view.camera
    az, el, dist = np.radians(cam.azimuth), np.radians(cam.elevation), cam.distance
    cam_pos = np.array([np.cos(el)*np.sin(az)*dist,
                        np.sin(el)*dist,
                        np.cos(el)*np.cos(az)*dist], np.float32)
    nebula.update(t, cam_pos, NEBULA_STRENGTH, _current_nebula_scale)

    stars.update(dt)
    for i, b in enumerate(butterflies):
        impul = _player_impulse_vec() if (player_idx is not None and i == player_idx) else None
        b.step(dt, stars.items, player_impulse=impul)

    newborns = []
    for b in butterflies:
        _, spawn = b.try_consume_star(stars.items)
        if spawn and len(butterflies)+len(newborns) < MAX_BUTTERFLIES:
            newborns.append(spawn_butterfly_near(b))

    if len(butterflies) >= 2 and len(butterflies) < MAX_BUTTERFLIES:
        centers = np.array([b.center4[:3] for b in butterflies], np.float32)
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(centers[i]-centers[j]) <= COLLISION_RADIUS:
                    if _rng.uniform() < COLLISION_BIRTH_PROB*dt*60:
                        newborns.append(spawn_butterfly_near(butterflies[i]))

    void_birth_acc += VOID_BIRTH_RATE*dt
    if void_birth_acc >= 1.0 and len(butterflies) < MAX_BUTTERFLIES:
        newborns.append(spawn_butterfly_void()); void_birth_acc -= 1.0

    if newborns: butterflies.extend(newborns)
    butterflies = [b for b in butterflies if b.alive()]
    if player_idx is not None and butterflies:
        player_idx = min(player_idx, len(butterflies)-1)
    elif not butterflies:
        player_idx = None

    # ----------- РЕНДЕР Бабочек -----------
    if butterflies:
        pts_all, cols_all = [], []
        halo_pts_all, halo_cols_all = [], []
        sin_term = 0.1*np.sin(0.35*t)

        for i, b in enumerate(butterflies):
            flap = 0.5*(1.0 - np.cos(b.flap_phase))
            P = (BASE_SHAPE * b.scale).copy()
            P[:,0] *= 1.0 + WING_AMP*(flap - 0.5)
            P[:,1] *= 1.0 - WING_AMP*(flap - 0.5)
            P[:,3] *= (1.0 + 0.25*(flap - 0.5))
            rotate_plane(P,0,3,b.angle1); rotate_plane(P,1,2,b.angle2)
            P += b.center4
            P3 = project_4d_to_3d(P, d=3.5)

            w_now  = P[:,3]
            w_norm = (w_now - w_now.min()) / max(1e-6, (w_now.max()-w_now.min()))
            if b.palette is None:
                hue = (w_norm + b.hue0 + sin_term) % 1.0
                sat = np.full_like(hue, b.sat, np.float32)
                val = np.full_like(hue, b.val, np.float32)
                rgb = hsv_to_rgb(hue, sat, val)
                alpha_val = MIN_ALPHA + (1.0 - MIN_ALPHA)*b.vital*b.fade_in
                cols = np.concatenate([rgb, np.full((rgb.shape[0],1), alpha_val, np.float32)], 1)
            else:
                rgba = b.palette.map(w_norm[:, None]).astype(np.float32)
                rgba[:,3] = MIN_ALPHA + (1.0 - MIN_ALPHA)*b.vital*b.fade_in
                cols = rgba

            # подсветим «игрока»
            if player_idx is not None and i == player_idx:
                cols[:, :3] = np.clip(cols[:, :3]*1.25, 0, 1)

            halo_cols = cols.copy(); halo_cols[:,3] = HALO_ALPHA * cols[:,3]
            pts_all.append(P3.astype(np.float32)); cols_all.append(cols)
            halo_pts_all.append(P3.astype(np.float32)); halo_cols_all.append(halo_cols)

            # трейл
            idx = (np.arange(len(b.trail3)) + b.trail_idx) % len(b.trail3)
            trail = b.trail3[idx]
            alphas = np.linspace(0.0, 1.0, len(trail), dtype=np.float32)
            alphas = (alphas**1.5) * (0.25 + 0.75*b.vital*b.fade_in)
            if b.palette is None:
                hue0 = (0.5 + b.hue0 + sin_term) % 1.0
                rgb0 = hsv_to_rgb(np.array([hue0],np.float32),
                                  np.array([b.sat],np.float32),
                                  np.array([b.val],np.float32))[0]
            else:
                rgb0 = b.palette.map(np.array([[0.6]], np.float32))[0,:3]
            core_cols = np.column_stack([np.tile(rgb0, (len(trail),1)), alphas]).astype(np.float32)
            halo_cols_tr = core_cols.copy(); halo_cols_tr[:,3] *= HALO_ALPHA*1.6
            b.line.set_data(pos=trail, color=core_cols, width=_current_trail_w)
            b.halo.set_data(pos=trail, color=halo_cols_tr, width=_current_trail_w*GLOW_LINE_SCALE)

        pts  = np.vstack(pts_all); cols = np.vstack(cols_all)
        hpts = np.vstack(halo_pts_all); hcls = np.vstack(halo_cols_all)
        butter_halo_viz.set_data(hpts, face_color=hcls,
                                 size=_current_marker_size*GLOW_MARKER_SCALE, edge_width=0.0)
        butter_viz.set_data(pts, face_color=cols,
                            size=_current_marker_size, edge_width=0.0)
    else:
        butter_viz.set_data(np.zeros((1,3),np.float32), face_color=(0,0,0,0), size=0.1)
        butter_halo_viz.set_data(np.zeros((1,3),np.float32), face_color=(0,0,0,0), size=0.1)

    # ----------- РЕНДЕР Звёзд -----------
    if stars.items:
        S = np.stack([s.pos3 for s in stars.items],0).astype(np.float32)
        d = np.linalg.norm(S, axis=1)
        base = _current_star_base * (1.6/(1.0 + 0.35*d))
        pulse = 1.0 + 0.25*np.sin(t*2.2 + np.linspace(0, 6.28318, len(S), dtype=np.float32))
        size_core = base * pulse; size_halo = size_core * 2.0
        tw = (np.sin(t*2.0 + np.linspace(0, 6.28318, len(S), dtype=np.float32)) * 0.5 + 0.5)
        tw = 0.7 + 0.3*tw
        c_base = np.stack([1.0*tw, 0.9*tw, 0.55*tw],1)
        jitter = np.clip(c_base + (np.random.uniform(-0.06,0.06,c_base.shape).astype(np.float32)), 0, 1)
        star_colors = np.concatenate([jitter, np.ones((len(S),1),np.float32)],1)
        halo_colors = star_colors.copy(); halo_colors[:,3] *= 0.35
        stars_core_viz.set_data(S, face_color=star_colors, size=size_core.astype(np.float32),
                                edge_width=0.0, symbol='disc')
        stars_halo_viz.set_data(S, face_color=halo_colors, size=size_halo.astype(np.float32),
                                edge_width=0.0, symbol='disc')

        seg_pos, seg_col = [], []
        for i, st in enumerate(stars.items):
            c = st.pos3; col = star_colors[i, :3]; a = 0.85 * star_colors[i, 3]
            r_out = (size_core[i] * 0.018); r_in  = r_out * SPIKE_INNER_FR
            r_out *= (1.0 + 0.15*np.sin(t*2.2 + i*0.7))
            for dvec in st.dirs:
                p0 = c + dvec * r_in; p1 = c + dvec * r_out
                seg_pos += [p0, p1]; seg_col += [[col[0], col[1], col[2], a], [col[0], col[1], col[2], a]]
        if seg_pos:
            seg_pos = np.asarray(seg_pos, dtype=np.float32)
            seg_col = np.asarray(seg_col, dtype=np.float32)
            connect = np.arange(seg_pos.shape[0], dtype=np.uint32).reshape(-1,2)
            stars_spikes_viz.set_data(pos=seg_pos, color=seg_col,
                                      width=_current_spikes_w, connect=connect)
        else:
            stars_spikes_viz.set_data(pos=np.zeros((2,3),np.float32),
                                      color=np.zeros((2,4),np.float32),
                                      width=_current_spikes_w, connect=np.array([[0,1]],np.uint32))
    else:
        stars_core_viz.set_data(np.zeros((1,3),np.float32), face_color=(0,0,0,0), size=0.1)
        stars_halo_viz.set_data(np.zeros((1,3),np.float32), face_color=(0,0,0,0), size=0.1)
        stars_spikes_viz.set_data(pos=np.zeros((2,3),np.float32),
                                  color=np.zeros((2,4),np.float32),
                                  width=_current_spikes_w, connect=np.array([[0,1]],np.uint32))

    # ----------- HUD -----------
    if hud_on:
        life = butterflies[player_idx].vital if (player_idx is not None and butterflies) else 0.0
        life_pct = int(round(life*100))
        hud_text.text = f"Бабочка: {player_idx if player_idx is not None else '-'} | Жизнь: {life_pct}%   (TAB — смена, WASD/QE — импульсы, Shift — ускорение, H — HUD, 1/2 — туман)"
        # ширина заполнения
        fill_w = max(2.0, 172.0 * np.clip(life, 0.0, 1.0))
        # центр fill смещаем так, чтобы «заливка» росла слева направо внутри рамки
        hud_fill.width = fill_w
        hud_fill.center = (34 + fill_w/2, 42)

# таймер
timer = app.Timer(interval='auto', connect=update, start=True)

if __name__ == '__main__':
    app.run()
