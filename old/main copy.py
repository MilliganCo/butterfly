import numpy as np
from vispy import app, scene, color

# ---------- 4D математика ----------
def rotate_plane(P, i, j, theta):
    """Повернуть точки P (N,4) в плоскости (i,j) на угол theta (SO(4) элементарная ротация)."""
    c, s = np.cos(theta), np.sin(theta)
    Pi, Pj = P[:, i].copy(), P[:, j].copy()
    P[:, i] = c * Pi - s * Pj
    P[:, j] = s * Pi + c * Pj
    return P

def project_4d_to_3d(P4, d=3.5):
    """
    Простейшая перспективная проекция 4D->3D вдоль оси w:
      X' = X * d / (d - w),  где X = (x,y,z)
    d выбирай > max(w) для стабильности.
    """
    w = P4[:, 3]
    denom = (d - w)
    # защита от деления на ноль
    denom[np.abs(denom) < 1e-6] = 1e-6
    factor = (d / denom).reshape(-1, 1)
    return P4[:, :3] * factor

# ---------- Данные "бабочки" в R^4 ----------
def butterfly_4d(n=12000):
    """
    4D-кривая: берём известную 2D butterfly (в полярных)
      r = e^{cos t} - 2 cos(4t) + sin^5(t/12)
    и "выдавливаем" в z,w для красивой толщины/ритма.
    """
    t = np.linspace(0, 24 * np.pi, n).astype(np.float32)
    r = np.exp(np.cos(t)) - 2 * np.cos(4 * t) + np.sin(t / 12) ** 5
    x = r * np.sin(t)
    y = r * np.cos(t)
    z = 0.55 * np.sin(3 * t)            # «рифление»
    w = 0.55 * np.cos(2 * t)            # «дыхание» по четвёртой оси
    P4 = np.stack([x, y, z, w], axis=1).astype(np.float32)
    return P4, t

# ---------- Сцена ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1100, 800), show=True, bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=55, distance=8.0, elevation=22, azimuth=35)

markers = scene.visuals.Markers(parent=view.scene)
markers.antialias = 1

P0, t = butterfly_4d(16000)

# цвет по w (потом будет анимироваться)
cm = color.get_colormap('viridis')

angle1 = 0.0   # поворот в плоскости (x,w) -> (0,3)
angle2 = 0.0   # поворот в плоскости (y,z) -> (1,2)
speed1 = 0.9   # рад/с
speed2 = 0.6

def update(ev):
    global angle1, angle2
    dt = ev.dt if ev.dt is not None else 1.0 / 60.0
    angle1 += speed1 * dt
    angle2 += speed2 * dt

    # Ротации в 4D
    P = P0.copy()
    rotate_plane(P, 0, 3, angle1)  # (x,w)
    rotate_plane(P, 1, 2, angle2)  # (y,z)

    # Проекция 4D -> 3D
    P3 = project_4d_to_3d(P, d=3.5)

    # Цвет кодируем по текущему w после поворота
    w_now = P[:, 3]
    w_norm = (w_now - w_now.min()) / max(1e-6, (w_now.max() - w_now.min()))
    cols = cm.map(w_norm)

    # Обновление геометрии
    markers.set_data(P3, face_color=cols, size=2.0, edge_width=0.0)

timer = app.Timer(interval='auto', connect=update, start=True)

if __name__ == '__main__':
    app.run()
