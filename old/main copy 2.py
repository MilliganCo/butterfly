# -*- coding: utf-8 -*-
# butterfly_swarm_4d.py
# Рой 4D-бабочек с целями-"звёздочками", делением/угасанием и редкими спонтанными рождениями.
# Зависимости: numpy, vispy, и любой Qt-бэкенд (PySide6 РЕКОМЕНДУЕТСЯ; PyQt5 тоже ок).

import os
import math
import time
import numpy as np

# --- (опционально) помогем Qt найти платформенные плагины на Windows ---
# и явно попросим vispy использовать доступный Qt-бэкенд.
try:
    import PySide6  # предпочтительно для Python 3.12
    from vispy import app
    app.use_app('pyside6')
    pyside_dir = os.path.dirname(PySide6.__file__)
    qt_plugins = os.path.join(pyside_dir, "plugins", "platforms")
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", qt_plugins)
except Exception:
    try:
        import PyQt5
        from vispy import app
        app.use_app('pyqt5')
    except Exception:
        from vispy import app  # VisPy выберет другой доступный бэкенд

# Лёгкая защита от странных DPI в Windows (можно убрать, если не нужно)
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from vispy import scene, color

# -------------------- 4D математика --------------------
def rotate_plane(P, i, j, theta):
    """Элементарное вращение в плоскости (i,j) для массива точек P (N,4)."""
    c, s = np.cos(theta), np.sin(theta)
    Pi, Pj = P[:, i].copy(), P[:, j].copy()
    P[:, i] = c * Pi - s * Pj
    P[:, j] = s * Pi + c * Pi * 0.0 + s * Pi * 0.0 + s * Pj  # защита от aliasing
    P[:, j] = s * Pi + c * Pj
    return P

def project_4d_to_3d(P4, d=3.5):
    """Перспективная проекция 4D->3D вдоль оси w: X' = X * d / (d - w)."""
    w = P4[:, 3]
    denom = (d - w)
    denom[np.abs(denom) < 1e-6] = 1e-6
    factor = (d / denom).reshape(-1, 1)
    return P4[:, :3] * factor

def butterfly_4d(n=2000, scale=0.25):
    """Базовая форма 4D-бабочки как кривой, выдавленной в z,w."""
    t = np.linspace(0, 24 * np.pi, n).astype(np.float32)
    r = np.exp(np.cos(t)) - 2 * np.cos(4 * t) + np.sin(t / 12) ** 5
    x = r * np.sin(t)
    y = r * np.cos(t)
    z = 0.55 * np.sin(3 * t)
    w = 0.55 * np.cos(2 * t)
    P4 = np.stack([x, y, z, w], axis=1).astype(np.float32)
    # нормируем по длине 3D-части и масштабируем
    P4 /= (np.max(np.linalg.norm(P4[:, :3], axis=1)) + 1e-6)
    P4 *= scale
    return P4, t

def hsv_to_rgb(h, s, v):
    """Векторная конвертация HSV->RGB для массивов h,s,v в [0,1]."""
    i = np.floor(h*6).astype(np.int32)
    f = h*6 - i
    p = v*(1-s)
    q = v*(1-f*s)
    t = v*(1-(1-f)*s)
    i_mod = i % 6
    out = np.zeros((h.shape[0], 3), dtype=np.float32)
    m = (i_mod==0); out[m] = np.stack([v[m], t[m], p[m]], axis=1)
    m = (i_mod==1); out[m] = np.stack([q[m], v[m], p[m]], axis=1)
    m = (i_mod==2); out[m] = np.stack([p[m], v[m], t[m]], axis=1)
    m = (i_mod==3); out[m] = np.stack([p[m], q[m], v[m]], axis=1)
    m = (i_mod==4); out[m] = np.stack([t[m], p[m], v[m]], axis=1)
    m = (i_mod==5); out[m] = np.stack([v[m], p[m], q[m]], axis=1)
    return out

# -------------------- параметры симуляции --------------------
MAX_BUTTERFLIES      = 40
INIT_BUTTERFLIES     = 6
POINTS_PER_BUTTERFLY = 2400

STAR_MAX        = 28
STAR_SPAWN_RATE = 0.8       # звёзд/сек в среднем
STAR_TTL        = (6.0, 12.0)
STAR_RADIUS     = 0.22

SPLIT_BOOST     = 0.45      # «жизнь» при съеденной звезде
FADE_RATE       = 0.045     # убывание «жизни»/сек без цели
BASE_FADE       = 0.01      # базовое убывание/сек всегда

COLLISION_RADIUS     = 0.25
COLLISION_BIRTH_PROB = 0.05  # малая вероятность рождения при столкновении
VOID_BIRTH_RATE      = 0.03  # редкие рождения «из пустоты», в среднем/сек

OU_BETA     = 0.8
OU_SIGMA    = 0.9
SPEED_LIMIT = 0.9
SEEK_GAIN   = 1.6
W_OSC       = 1.2           # «дыхание» по w

# визуальные базовые размеры (в пикселях при высоте окна 900)
BASE_MARKER_SIZE = 1.8
BASE_STAR_SIZE   = 6.5

BG_COLOR = 'black'

# -------------------- общие формы --------------------
BASE_SHAPE, _t = butterfly_4d(POINTS_PER_BUTTERFLY, scale=0.22)

# -------------------- сущности --------------------
_rng = np.random.default_rng()

class Star:
    __slots__ = ("pos3", "ttl", "born", "alive")
    def __init__(self):
        self.pos3 = _rng.uniform(-2.8, 2.8, size=3).astype(np.float32)
        self.ttl  = float(_rng.uniform(*STAR_TTL))
        self.born = time.perf_counter()
        self.alive = True
    def update(self, dt):
        self.ttl -= dt
        if self.ttl <= 0:
            self.alive = False

class Butterfly:
    __slots__ = ("center4", "vel4", "angle1", "angle2", "hue0",
                 "vital", "fade_in", "target_id", "scale")
    def __init__(self, center4=None, hue0=None, copy_from=None):
        if copy_from is not None:
            self.center4 = copy_from.center4.copy()
            self.vel4    = copy_from.vel4.copy()*0.0
            self.angle1  = copy_from.angle1 + float(_rng.uniform(-0.5, 0.5))
            self.angle2  = copy_from.angle2 + float(_rng.uniform(-0.5, 0.5))
            self.hue0    = (copy_from.hue0 + float(_rng.uniform(0.05, 0.15))) % 1.0
            self.vital   = min(1.0, copy_from.vital*0.7 + 0.3)
            self.fade_in = 0.0
            self.scale   = copy_from.scale * float(_rng.uniform(0.95, 1.05))
        else:
            self.center4 = (_rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
                            if center4 is None else center4.astype(np.float32))
            self.vel4    = _rng.normal(0, 0.3, size=4).astype(np.float32)
            self.angle1  = float(_rng.uniform(0, 2*np.pi))
            self.angle2  = float(_rng.uniform(0, 2*np.pi))
            self.hue0    = float(_rng.uniform(0, 1.0) if hue0 is None else hue0)
            self.vital   = 0.35
            self.fade_in = 0.0
            self.scale   = np.float32(1.0)
        self.target_id = -1

    def pick_target(self, stars3):
        if not stars3:
            self.target_id = -1
            return
        C3 = self.center4[:3]
        dists = [np.linalg.norm(C3 - s.pos3) for s in stars3]
        self.target_id = int(np.argmin(dists)) if len(dists) else -1

    def step(self, dt, stars3):
        # «дыхание» по w
        self.center4[3] += 0.2*np.sin(W_OSC*time.perf_counter())*dt

        # актуализируем цель
        if self.target_id < 0 or self.target_id >= len(stars3) or not stars3[self.target_id].alive:
            self.pick_target(stars3)

        # ОУ-шум для скорости
        noise = _rng.normal(0.0, 1.0, size=4).astype(np.float32)
        self.vel4 += (-OU_BETA*self.vel4)*dt + (OU_SIGMA*np.sqrt(max(dt,1e-3))*noise)

        # стягиваемся к цели по (x,y,z)
        if 0 <= self.target_id < len(stars3) and stars3[self.target_id].alive:
            tgt = stars3[self.target_id].pos3
            dir3 = (tgt - self.center4[:3])
            self.vel4[:3] += SEEK_GAIN * dir3 * dt

        # ограничение скорости
        vnorm = np.linalg.norm(self.vel4)
        if vnorm > SPEED_LIMIT:
            self.vel4 *= (SPEED_LIMIT / (vnorm + 1e-6))

        # шаг положения
        self.center4 += self.vel4 * dt

        # вращения формы
        self.angle1 += 0.9*dt
        self.angle2 += 0.6*dt

        # «жизнь»
        self.vital = max(0.0, self.vital - BASE_FADE*dt - (FADE_RATE*dt if self.target_id < 0 else 0.0))
        self.fade_in = min(1.0, self.fade_in + 0.8*dt)

    def try_consume_star(self, stars3):
        if not (0 <= self.target_id < len(stars3)):
            return False
        star = stars3[self.target_id]
        if not star.alive:
            return False
        dist = np.linalg.norm(self.center4[:3] - star.pos3)
        if dist <= STAR_RADIUS:
            star.alive = False
            self.vital = min(1.0, self.vital + SPLIT_BOOST)
            return True
        return False

    def alive(self):
        return self.vital > 0.02

# -------------------- сцена VisPy --------------------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True, bgcolor=BG_COLOR)
view   = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=55, distance=7.5, elevation=22, azimuth=35)

butter_viz = scene.visuals.Markers(parent=view.scene)
butter_viz.antialias = 1
stars_viz  = scene.visuals.Markers(parent=view.scene)
stars_viz.antialias = 1

# динамически масштабируемые размеры (px). Базовый масштаб — на высоту 900.
_current_marker_size = BASE_MARKER_SIZE
_current_star_size   = BASE_STAR_SIZE

def _apply_scale_from_canvas():
    """Скалируем размеры маркеров пропорционально высоте окна."""
    global _current_marker_size, _current_star_size
    _, h = canvas.size
    scale = max(0.33, min(3.0, h / 900.0))  # ограничим разумные пределы
    _current_marker_size = BASE_MARKER_SIZE * scale
    _current_star_size   = BASE_STAR_SIZE   * scale

_apply_scale_from_canvas()

@canvas.events.resize.connect
def _on_resize(event):
    _apply_scale_from_canvas()

# -------------------- симуляция --------------------
butterflies = [Butterfly() for _ in range(INIT_BUTTERFLIES)]

class StarManager:
    def __init__(self):
        self.items = [Star() for _ in range(10)]
        self.spawn_acc = 0.0
    def update(self, dt):
        # спавн новых
        self.spawn_acc += STAR_SPAWN_RATE * dt
        while self.spawn_acc >= 1.0 and len(self.items) < STAR_MAX:
            self.items.append(Star())
            self.spawn_acc -= 1.0
        # апдейты и чистка
        alive = []
        for s in self.items:
            if s.alive:
                s.update(dt)
                if s.alive:
                    alive.append(s)
        self.items = alive

stars = StarManager()

last_time = time.perf_counter()
void_birth_acc = 0.0

def spawn_butterfly_near(parent: Butterfly):
    b = Butterfly(copy_from=parent)
    b.center4[:3] += _rng.normal(0, 0.05, size=3).astype(np.float32)
    b.center4[3]  += _rng.normal(0, 0.03)
    return b

def spawn_butterfly_void():
    c4 = _rng.uniform(-1.4, 1.4, size=4).astype(np.float32)
    b = Butterfly(center4=c4, hue0=float(_rng.uniform(0,1.0)))
    b.vital = 0.0
    b.fade_in = 0.0
    return b

def update(ev):
    global last_time, void_birth_acc, butterflies
    t = time.perf_counter()
    dt = min(0.05, max(1e-4, t - last_time))
    last_time = t

    # звёзды
    stars.update(dt)

    # бабочки шаг
    for b in butterflies:
        b.step(dt, stars.items)

    # цели / деление
    newborns = []
    for b in butterflies:
        if b.try_consume_star(stars.items):
            if len(butterflies) + len(newborns) < MAX_BUTTERFLIES:
                newborns.append(spawn_butterfly_near(b))

    # столкновения -> шанс рождения
    if len(butterflies) >= 2 and len(butterflies) < MAX_BUTTERFLIES:
        centers = np.array([b.center4[:3] for b in butterflies], dtype=np.float32)
        n = len(centers)
        for i in range(n):
            ci = centers[i]
            for j in range(i+1, n):
                if np.linalg.norm(ci - centers[j]) <= COLLISION_RADIUS:
                    # нормируем к шанс/сек
                    if _rng.uniform() < COLLISION_BIRTH_PROB * dt * 60:
                        newborns.append(spawn_butterfly_near(butterflies[i]))

    # спонтанные рождения
    void_birth_acc += VOID_BIRTH_RATE * dt
    if void_birth_acc >= 1.0 and len(butterflies) < MAX_BUTTERFLIES:
        newborns.append(spawn_butterfly_void())
        void_birth_acc -= 1.0

    # добавим новорождённых
    if newborns:
        butterflies.extend(newborns)

    # удалить потухших
    butterflies = [b for b in butterflies if b.alive()]

    # ---------- РЕНДЕР ----------
    # Бабочки
    if butterflies:
        pts_list = []
        col_list = []
        sin_term = 0.1*np.sin(0.5*t)
        for b in butterflies:
            P = (BASE_SHAPE * b.scale).copy()
            rotate_plane(P, 0, 3, b.angle1)  # (x,w)
            rotate_plane(P, 1, 2, b.angle2)  # (y,z)
            P += b.center4  # перенос в центр бабочки

            # цвет — перелив по w + индивидуальный hue
            w_now  = P[:, 3]
            w_norm = (w_now - w_now.min()) / max(1e-6, (w_now.max() - w_now.min()))
            hue = (w_norm + b.hue0 + sin_term) % 1.0
            sat = np.full_like(hue, 0.85, dtype=np.float32)
            val = np.full_like(hue, 0.95, dtype=np.float32)

            rgb = hsv_to_rgb(hue, sat, val)  # (N,3)
            alpha_val = 0.15 + 0.85 * b.vital * b.fade_in
            alpha_arr = np.full((rgb.shape[0], 1), alpha_val, dtype=np.float32)
            cols = np.concatenate([rgb, alpha_arr], axis=1).astype(np.float32)

            P3 = project_4d_to_3d(P, d=3.5)

            pts_list.append(P3)
            col_list.append(cols)

        pts  = np.vstack(pts_list).astype(np.float32)
        cols = np.vstack(col_list).astype(np.float32)
        butter_viz.set_data(pts, face_color=cols, size=_current_marker_size, edge_width=0.0)
    else:
        butter_viz.set_data(np.zeros((1,3),dtype=np.float32), face_color=(0,0,0,0), size=0.1)

    # Звёзды
    if stars.items:
        S = np.stack([s.pos3 for s in stars.items], axis=0).astype(np.float32)
        # мягкое мерцание
        phases = np.linspace(0, 6.28318, len(stars.items), dtype=np.float32)
        tw = (np.sin(t*3.0 + phases) * 0.5 + 0.5)  # 0..1
        tw = 0.7 + 0.3*tw
        star_colors = np.stack([1.0*tw, 0.95*tw, 0.6*tw, np.ones_like(tw)], axis=1).astype(np.float32)
        stars_viz.set_data(S, face_color=star_colors, size=_current_star_size, edge_width=0.0)
    else:
        stars_viz.set_data(np.zeros((1,3),dtype=np.float32), face_color=(0,0,0,0), size=0.1)

# Таймер
timer = app.Timer(interval='auto', connect=update, start=True)

if __name__ == '__main__':
    app.run()
