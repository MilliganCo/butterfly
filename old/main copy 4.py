# -*- coding: utf-8 -*-
# butterfly_swarm_4d.py — рой 4D-бабочек с трейлами, "свечением" и физикой полёта.
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

# -------------------- 4D математика --------------------
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

# wander
WANDER_GAIN   = 0.35
WANDER_T_MIN  = 3.0
WANDER_T_MAX  = 6.0

# "бабочья" физика
FLAP_FREQ     = (4.0, 8.0)   # Гц
FLAP_FORCE    = 0.85         # импульс вдоль курса
LIFT_GAIN     = 0.35         # подъём при взмахе по z
GLIDE_DRAG    = 0.25         # демпфирование в глайде

# "взмах крыльев" — деформация формы
WING_EPS      = 0.14         # анизотропный масштаб по X/Y
W_OSC         = 1.0          # дыхание по w (плавно)

# визуал/свечение
BASE_MARKER_SIZE = 1.6     # px при высоте окна 900
BASE_STAR_SIZE   = 5.5
BASE_TRAIL_WIDTH = 1.4
GLOW_MARKER_SCALE = 2.4
GLOW_LINE_SCALE   = 2.6
HALO_ALPHA        = 0.18
BG_COLOR = 'black'

BASE_SHAPE, _t = butterfly_4d(POINTS_PER_BUTTERFLY, scale=0.20)
_rng = np.random.default_rng()

# -------------------- сущности --------------------
class Star:
    __slots__ = ("pos3","ttl","born","alive","hue_jitter")
    def __init__(self):
        r = _rng.uniform(0.3, 1.0)
        theta = _rng.uniform(0, 2*np.pi)
        phi = np.arccos(_rng.uniform(-1, 1))
        radius = 3.2 * r**1.6
        x = radius*np.sin(phi)*np.cos(theta)
        y = radius*np.sin(phi)*np.sin(theta)
        z = radius*np.cos(phi)
        self.pos3 = np.array([x, y, z], np.float32)
        self.ttl  = float(_rng.uniform(*STAR_TTL))
        self.born = time.perf_counter()
        self.alive = True
        self.hue_jitter = float(_rng.uniform(-0.03, 0.03))
    def update(self, dt):
        self.ttl -= dt
        if self.ttl <= 0:
            self.alive = False

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

        # цветовой профиль
        if _rng.random() < 0.5:
            self.palette = None
            self.hue0 = float(_rng.uniform(0, 1.0))
            self.sat  = float(_rng.uniform(0.65, 0.95))
            self.val  = float(_rng.uniform(0.80, 0.98))
        else:
            cmaps = ['viridis','plasma','magma','inferno','turbo',
                     'twilight','ice','winter','autumn','spring']
            self.palette = color.get_colormap(_rng.choice(cmaps))
            self.hue0 = self.sat = self.val = 0.0

        # крылья
        self.flap_phase = float(_rng.uniform(0, 2*np.pi))
        self.flap_omega = 2*np.pi*float(_rng.uniform(*FLAP_FREQ))

        # трейл
        self.trail3 = np.zeros((64, 3), np.float32)
        self.trail_idx = 0
        self.trail_timer = 0.0
        self.line = None   # visuals.Line
        self.halo = None   # visuals.Line (широкий)

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

    def step(self, dt, stars):
        tnow = time.perf_counter()
        self.center4[3] += 0.18*np.sin(W_OSC*tnow)*dt

        # режимы
        self.mode_ttl -= dt
        if self.mode_ttl <= 0:
            self.mode = 'wander' if self.mode == 'seek' else 'seek'
            self.mode_ttl = float(_rng.uniform(WANDER_T_MIN, WANDER_T_MAX))
            if self.mode == 'wander':
                self.wander_center = self.center4[:3] + _rng.normal(0, 0.6, 3).astype(np.float32)

        if self.mode == 'seek':
            if self.target_id < 0 or self.target_id >= len(stars) or not stars[self.target_id].alive:
                self.pick_target(stars)

        # ускорение
        self.acc4 *= 0.0
        self.acc4 += _rng.normal(0.0, OU_SIGMA, 4).astype(np.float32) * np.sqrt(max(dt,1e-3))
        self.acc4 += -OU_BETA*self.vel4

        # "взмах" — импульс вдоль курса + подъём
        self.flap_phase += self.flap_omega * dt
        flap = max(0.0, np.sin(self.flap_phase))  # только активная полуволна
        v3 = self.vel4[:3]
        vnorm = np.linalg.norm(v3) + 1e-6
        heading = v3 / vnorm
        self.acc4[:3] += FLAP_FORCE * flap * heading
        self.acc4[2]  += LIFT_GAIN  * flap

        # стремление
        if self.mode == 'seek' and 0 <= self.target_id < len(stars) and stars[self.target_id].alive:
            dir3 = (stars[self.target_id].pos3 - self.center4[:3])
            self.acc4[:3] += SEEK_GAIN * dir3
        else:
            want = self._wander_point(tnow) - self.center4[:3]
            self.acc4[:3] += WANDER_GAIN * want

        # ограничение ускорения / скорость / интеграция
        an = np.linalg.norm(self.acc4)
        if an > ACCEL_LIMIT:
            self.acc4 *= ACCEL_LIMIT/(an+1e-6)
        self.vel4 += self.acc4 * dt
        if self.mode != 'seek':  # в глайде сильнее демпфим
            self.vel4 *= (1.0 - GLIDE_DRAG*dt)
        vn = np.linalg.norm(self.vel4)
        if vn > SPEED_LIMIT:
            self.vel4 *= (SPEED_LIMIT/(vn+1e-6))
        self.center4 += self.vel4 * dt

        self.angle1 += 0.8*dt
        self.angle2 += 0.55*dt

        # жизнь
        self.vital = max(0.0, self.vital - BASE_FADE*dt - (FADE_RATE*dt if self.mode=='seek' and self.target_id<0 else 0.0))
        self.fade_in = min(1.0, self.fade_in + 0.8*dt)

        # трейл-сэмплинг (каждые ~0.03с)
        self.trail_timer += dt
        if self.trail_timer >= 0.03:
            self.trail_timer = 0.0
            c3 = project_point4_to_3(self.center4)
            self.trail3[self.trail_idx] = c3
            self.trail_idx = (self.trail_idx + 1) % len(self.trail3)

        # кулдауны
        if self.repro_cd > 0:
            self.repro_cd -= dt

    def try_consume_star(self, stars):
        if not (0 <= self.target_id < len(stars)):
            return False, False
        st = stars[self.target_id]
        if not st.alive: return False, False
        if np.linalg.norm(self.center4[:3] - st.pos3) <= STAR_RADIUS:
            st.alive = False
            self.vital = min(1.0, self.vital + SPLIT_BOOST)
            spawn = (self.repro_cd <= 0) and (_rng.random() < SPLIT_PROB)
            if spawn:
                self.repro_cd = float(_rng.uniform(*REPRO_COOLDOWN))
            return True, spawn
        return False, False

    def alive(self): return self.vital > 0.02

# -------------------- сцена --------------------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True, bgcolor=BG_COLOR)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=55, distance=7.5, elevation=22, azimuth=35)

# бабочки: «ядро» и «ореол» (свечение)
butter_viz = scene.visuals.Markers(parent=view.scene); butter_viz.antialias = 1
butter_halo_viz = scene.visuals.Markers(parent=view.scene); butter_halo_viz.antialias = 1
# звёзды
stars_viz = scene.visuals.Markers(parent=view.scene); stars_viz.antialias = 1

# трейлы — по 2 линии на бабочку (core + halo)
from vispy.scene import visuals

_current_marker_size = BASE_MARKER_SIZE
_current_star_base   = BASE_STAR_SIZE
_current_trail_w     = BASE_TRAIL_WIDTH

def _apply_scale_from_canvas():
    global _current_marker_size, _current_star_base, _current_trail_w
    _, h = canvas.size
    scale = max(0.33, min(3.0, h/900.0))
    _current_marker_size = BASE_MARKER_SIZE * scale
    _current_star_base   = BASE_STAR_SIZE   * scale
    _current_trail_w     = BASE_TRAIL_WIDTH * scale

_apply_scale_from_canvas()
@canvas.events.resize.connect
def _on_resize(event): _apply_scale_from_canvas()

# -------------------- симуляция --------------------
butterflies = [Butterfly() for _ in range(INIT_BUTTERFLIES)]

class StarManager:
    def __init__(self):
        self.items = [Star() for _ in range(10)]
        self.spawn_acc = 0.0
    def update(self, dt):
        self.spawn_acc += STAR_SPAWN_RATE * dt
        while self.spawn_acc >= 1.0 and len(self.items) < STAR_MAX:
            self.items.append(Star()); self.spawn_acc -= 1.0
        alive = []
        for s in self.items:
            if s.alive:
                s.update(dt)
                if s.alive: alive.append(s)
        self.items = alive

stars = StarManager()
last_time = time.perf_counter()
void_birth_acc = 0.0

def spawn_butterfly_near(parent):
    b = Butterfly(copy_from=parent)
    b.center4[:3] += _rng.normal(0, 0.05, 3).astype(np.float32)
    b.center4[3]  += _rng.normal(0, 0.03)
    # создать трейлы
    b.line = visuals.Line(parent=view.scene, method='gl')
    b.halo = visuals.Line(parent=view.scene, method='gl')
    return b

def spawn_butterfly_void():
    c4 = _rng.uniform(-1.2, 1.2, 4).astype(np.float32)
    b = Butterfly(center4=c4)
    b.vital = 0.0; b.fade_in = 0.0
    b.line = visuals.Line(parent=view.scene, method='gl')
    b.halo = visuals.Line(parent=view.scene, method='gl')
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

def ensure_trail_visuals(b: Butterfly):
    if b.line is None: b.line = visuals.Line(parent=view.scene, method='gl')
    if b.halo is None: b.halo = visuals.Line(parent=view.scene, method='gl')

def remove_trail_visuals(b: Butterfly):
    if b.line is not None: b.line.parent = None; b.line = None
    if b.halo is not None: b.halo.parent = None; b.halo = None

def update(ev):
    global last_time, void_birth_acc, butterflies
    t = time.perf_counter()
    dt = min(0.05, max(1e-4, t - last_time))
    last_time = t

    stars.update(dt)
    for b in butterflies:
        b.step(dt, stars.items)

    newborns = []
    for b in butterflies:
        consumed, spawn = b.try_consume_star(stars.items)
        if spawn and len(butterflies)+len(newborns) < MAX_BUTTERFLIES:
            newborns.append(spawn_butterfly_near(b))

    # столкновения
    if len(butterflies) >= 2 and len(butterflies) < MAX_BUTTERFLIES:
        centers = np.array([b.center4[:3] for b in butterflies], np.float32)
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(centers[i]-centers[j]) <= COLLISION_RADIUS:
                    if _rng.uniform() < COLLISION_BIRTH_PROB*dt*60:
                        newborns.append(spawn_butterfly_near(butterflies[i]))

    # спонтанные рождения
    void_birth_acc += VOID_BIRTH_RATE*dt
    if void_birth_acc >= 1.0 and len(butterflies) < MAX_BUTTERFLIES:
        newborns.append(spawn_butterfly_void()); void_birth_acc -= 1.0

    if newborns: butterflies.extend(newborns)

    # удалить потухших + их визуалы
    alive_list = []
    for b in butterflies:
        if b.alive():
            alive_list.append(b)
        else:
            remove_trail_visuals(b)
    butterflies = alive_list

    # ----------- РЕНДЕР -----------
    if butterflies:
        pts_all, cols_all = [], []
        halo_pts_all, halo_cols_all = [], []
        sin_term = 0.1*np.sin(0.45*t)

        for b in butterflies:
            # форма с "махами": анизотропный масштаб и пульс по w
            flap = max(0.0, np.sin(b.flap_phase))
            P = (BASE_SHAPE * b.scale).copy()
            # анизотропный масштаб по X/Y (крылья)
            sx = 1.0 + WING_EPS*flap
            sy = 1.0 - WING_EPS*flap
            P[:,0] *= sx; P[:,1] *= sy
            # пульс по w (микро-«дыхание» крыльев)
            P[:,3] *= (1.0 + 0.35*flap)

            rotate_plane(P,0,3,b.angle1); rotate_plane(P,1,2,b.angle2)
            P += b.center4
            P3 = project_4d_to_3d(P, d=3.5)

            # цвет
            w_now  = P[:,3]
            w_norm = (w_now - w_now.min()) / max(1e-6, (w_now.max()-w_now.min()))
            if b.palette is None:
                hue = (w_norm + b.hue0 + sin_term) % 1.0
                sat = np.full_like(hue, b.sat, np.float32)
                val = np.full_like(hue, b.val, np.float32)
                rgb = hsv_to_rgb(hue, sat, val)
                alpha_val = 0.15 + 0.85*b.vital*b.fade_in
                cols = np.concatenate([rgb, np.full((rgb.shape[0],1), alpha_val, np.float32)], 1)
            else:
                rgba = b.palette.map(w_norm[:, None]).astype(np.float32)  # FIX: [:,None]
                rgba[:,3] = 0.15 + 0.85*b.vital*b.fade_in
                cols = rgba

            # halo для маркеров (мягкое "свечение")
            halo_cols = cols.copy()
            halo_cols[:,3] = HALO_ALPHA * cols[:,3]

            pts_all.append(P3.astype(np.float32))
            cols_all.append(cols)
            halo_pts_all.append(P3.astype(np.float32))
            halo_cols_all.append(halo_cols)

            # --- трейл ---
            ensure_trail_visuals(b)
            # берём точки в правильном порядке из кольцевого буфера
            idx = (np.arange(len(b.trail3)) + b.trail_idx) % len(b.trail3)
            trail = b.trail3[idx]
            # альфа-градиент вдоль хвоста
            alphas = np.linspace(0.0, 1.0, len(trail), dtype=np.float32)
            alphas = (alphas**1.5) * (0.25 + 0.75*b.vital*b.fade_in)
            # цвет — средний цвет бабочки (по текущему колормэпу)
            if b.palette is None:
                # возьмём типичный цвет по середине
                hue0 = (0.5 + b.hue0 + sin_term) % 1.0
                rgb0 = hsv_to_rgb(np.array([hue0],np.float32),
                                  np.array([b.sat],np.float32),
                                  np.array([b.val],np.float32))[0]
            else:
                rgb0 = b.palette.map(np.array([[0.6]], np.float32))[0,:3]
            core_cols = np.column_stack([np.tile(rgb0, (len(trail),1)), alphas]).astype(np.float32)
            halo_cols_tr = core_cols.copy(); halo_cols_tr[:,3] *= HALO_ALPHA*1.6

            b.line.set_data(pos=trail, color=core_cols, width=_current_trail_w, method='gl')
            b.halo.set_data(pos=trail, color=halo_cols_tr, width=_current_trail_w*GLOW_LINE_SCALE, method='gl')

        pts  = np.vstack(pts_all)
        cols = np.vstack(cols_all)
        hpts = np.vstack(halo_pts_all)
        hcls = np.vstack(halo_cols_all)

        # сначала «ореол», затем «ядро» — для мягкого свечения
        butter_halo_viz.set_data(hpts, face_color=hcls,
                                 size=_current_marker_size*GLOW_MARKER_SCALE, edge_width=0.0)
        butter_viz.set_data(pts, face_color=cols,
                            size=_current_marker_size, edge_width=0.0)
    else:
        butter_viz.set_data(np.zeros((1,3),np.float32), face_color=(0,0,0,0), size=0.1)
        butter_halo_viz.set_data(np.zeros((1,3),np.float32), face_color=(0,0,0,0), size=0.1)

    # звёзды (размер по глубине + лёгкое мерцание)
    if stars.items:
        S = np.stack([s.pos3 for s in stars.items],0).astype(np.float32)
        d = np.linalg.norm(S, axis=1)
        size = _current_star_base * (1.6/(1.0 + 0.35*d))
        phases = np.linspace(0, 6.28318, len(stars.items), dtype=np.float32)
        tw = (np.sin(t*2.6 + phases) * 0.5 + 0.5)
        tw = 0.7 + 0.3*tw
        c_base = np.stack([1.0*tw, 0.9*tw, 0.55*tw],1)
        jitter = np.clip(c_base + (_rng.uniform(-0.06,0.06,c_base.shape).astype(np.float32)), 0, 1)
        star_colors = np.concatenate([jitter, np.ones((len(S),1),np.float32)],1)
        stars_viz.set_data(S, face_color=star_colors, size=size.astype(np.float32), edge_width=0.0)
    else:
        stars_viz.set_data(np.zeros((1,3),np.float32), face_color=(0,0,0,0), size=0.1)

# таймер
timer = app.Timer(interval='auto', connect=update, start=True)

if __name__ == '__main__':
    app.run()
