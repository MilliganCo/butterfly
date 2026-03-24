# -*- coding: utf-8 -*-
# 4D Butterflies — ModernGL (lines+ribbons)
# Бабочки как полилинии (экранная толщина) + «шипастые» звезды.
# Трейлы (accumulation), лёгкий glow и процедурная туманность.
# Управление:
#   ЛКМ+drag — орбита, колесо — зум,
#   TAB — следующая бабочка (режим игрока), H — HUD,
#   WASDQE — импульсы (в режиме игрока), Shift — ускорение,
#   SPACE — очистка трейлов,
#   -/= — длина хвоста (скорость затухания), 1/2 — сила туманности.
#
# pip install moderngl moderngl-window pyrr numpy

import math, time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import moderngl as mgl
import moderngl_window as mglw
from pyrr import Matrix44, Vector3

# ------------------------ ПАРАМЕТРЫ ------------------------
MAX_BUTTERFLIES      = 20
INIT_BUTTERFLIES     = 6
POINTS_PER_BUTTERFLY = 1400
DOWNSAMPLE_STEP      = 3
WING_AMP             = 0.55
FLAP_FREQ_RANGE      = (0.26, 0.34)

SPEED_LIMIT   = 0.75
ACCEL_LIMIT   = 1.0
OU_BETA       = 0.85
OU_SIGMA      = 0.50
SEEK_GAIN     = 1.0
WANDER_GAIN   = 0.28
LIFT_GAIN     = 0.28
FLAP_PUSH     = 0.70
W_OSC         = 0.8

# Боиды
NEIGHBOR_R   = 0.9
SEPARATION_W = 0.9
ALIGNMENT_W  = 0.25
COHESION_W   = 0.30

# Жизнь/события
LIFE_DECAY_S     = 0.010
LIFE_FROM_STAR   = 0.18
REPRO_REQ        = 0.25
REPRO_COST       = 0.20
STAR_RADIUS      = 0.22
VOID_BIRTH_RATE  = 0.004
COLLISION_R      = 0.20

# Звёзды
STAR_MAX        = 28
STAR_SPAWN_RATE = 0.55
STAR_TTL        = (7.0, 13.0)
STAR_SPIKES_MIN = 12
STAR_SPIKES_MAX = 18
SPIKE_INNER_FR  = 0.35
STAR_BASE_PX    = 9.0

# Рендер/Glow
ACCUM_DECAY       = 0.92
NEBULA_STRENGTH   = 0.18
FOV               = 55.0

# Толщины линий (px)
LINE_W_BFLY       = (2.2, 4.2)
LINE_W_SPIKE      = 2.0

# Видимость
FADEIN_SPEED      = 1.6
MIN_VIS_ALPHA     = 0.25

rng = np.random.default_rng()

# ------------------------ ГЕОМЕТРИЯ 4D ---------------------
def rotate_plane(P: np.ndarray, i: int, j: int, theta: float):
    c, s = math.cos(theta), math.sin(theta)
    Pi = P[:, i].copy()
    Pj = P[:, j].copy()
    P[:, i] = c * Pi - s * Pj
    P[:, j] = s * Pi + c * Pj
    return P

def project_4d_to_3d(P4: np.ndarray, d: float = 3.5) -> np.ndarray:
    w = P4[:, 3]
    denom = (d - w)
    denom[np.abs(denom) < 1e-6] = 1e-6
    factor = (d / denom).reshape(-1, 1)
    return P4[:, :3] * factor

def butterfly_curve_4d(n=POINTS_PER_BUTTERFLY, scale=0.22):
    t = np.linspace(0, 24*np.pi, n).astype(np.float32)
    r = np.exp(np.cos(t)) - 2*np.cos(4*t) + np.sin(t/12)**5
    x = r*np.sin(t); y = r*np.cos(t)
    z = 0.55*np.sin(3*t); w = 0.55*np.cos(2*t)
    P4 = np.stack([x, y, z, w], axis=1).astype(np.float32)
    P4 /= (np.max(np.linalg.norm(P4[:, :3], axis=1)) + 1e-6)
    P4 *= scale
    return P4

BASE_SHAPE_4D = butterfly_curve_4d()

# ------------------------ СИМУЛЯЦИЯ -----------------------
@dataclass
class Star:
    pos3: np.ndarray
    ttl: float
    born: float
    dirs: np.ndarray
    alive: bool = True
    @staticmethod
    def make():
        r = rng.uniform(0.3, 1.0)
        theta = rng.uniform(0, 2*np.pi)
        phi = math.acos(rng.uniform(-1, 1))
        radius = 3.2 * (r ** 1.6)
        x = radius*math.sin(phi)*math.cos(theta)
        y = radius*math.sin(phi)*math.sin(theta)
        z = radius*math.cos(phi)
        k = rng.integers(STAR_SPIKES_MIN, STAR_SPIKES_MAX+1)
        u = rng.normal(0, 1, size=(k,3)).astype(np.float32)
        u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-6)
        return Star(np.array([x,y,z], np.float32),
                    float(rng.uniform(*STAR_TTL)),
                    time.perf_counter(), u)
    def update(self, dt: float):
        self.ttl -= dt
        if self.ttl <= 0: self.alive = False

@dataclass
class Butterfly:
    center4: np.ndarray
    vel4: np.ndarray
    acc4: np.ndarray
    angle1: float
    angle2: float
    scale: float
    life:  float = 0.6
    fade_in: float = 0.0
    target: int = -1
    flap_phase: float = field(default_factory=lambda: rng.uniform(0, 2*np.pi))
    flap_omega: float = field(default_factory=lambda: 2*np.pi*rng.uniform(*FLAP_FREQ_RANGE))
    hue0: float = field(default_factory=lambda: rng.uniform(0,1))
    sat:  float = field(default_factory=lambda: rng.uniform(0.70, 0.95))
    val:  float = field(default_factory=lambda: rng.uniform(0.82, 0.98))
    mode: str = field(default_factory=lambda: 'seek' if rng.random() < 0.6 else 'wander')
    mode_ttl: float = field(default_factory=lambda: rng.uniform(3.0, 6.0))
    wander_center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    repro_cd: float = field(default_factory=lambda: rng.uniform(2.0, 4.0))
    def alive(self): return self.life > 0.02
    @staticmethod
    def make(center4=None, copy=None):
        if copy is not None:
            c = copy.center4.copy()
            s = copy.scale * rng.uniform(0.94, 1.06)
            v = np.zeros(4, np.float32)
            a = np.zeros(4, np.float32)
            b = Butterfly(c, v, a,
                          copy.angle1+rng.uniform(-0.25,0.25),
                          copy.angle2+rng.uniform(-0.25,0.25),
                          s, life=min(1.0, copy.life*0.75+0.25))
            b.wander_center = b.center4[:3].copy()
            b.flap_phase = copy.flap_phase + rng.uniform(-0.5,0.5)
            return b
        c = rng.uniform(-1.0,1.0,4).astype(np.float32) if center4 is None else center4.astype(np.float32)
        v = rng.normal(0, 0.25, 4).astype(np.float32)
        a = np.zeros(4, np.float32)
        b = Butterfly(c, v, a, rng.uniform(0,2*np.pi), rng.uniform(0,2*np.pi), 1.0)
        b.wander_center = b.center4[:3].copy()
        return b

def hsv_to_rgb_np(h, s, v):
    i = np.floor(h*6).astype(np.int32)
    f = h*6 - i
    p = v*(1-s); q = v*(1-f*s); t = v*(1-(1-f)*s); i%=6
    out = np.zeros((h.shape[0],3),np.float32)
    m=(i==0); out[m]=np.stack([v[m],t[m],p[m]],1)
    m=(i==1); out[m]=np.stack([q[m],v[m],p[m]],1)
    m=(i==2); out[m]=np.stack([p[m],v[m],t[m]],1)
    m=(i==3); out[m]=np.stack([p[m],q[m],v[m]],1)
    m=(i==4); out[m]=np.stack([t[m],p[m],v[m]],1)
    m=(i==5); out[m]=np.stack([v[m],p[m],q[m]],1)
    return out

# ------------------------ APP ------------------------------
class ButterflyApp(mglw.WindowConfig):
    title = "4D Butterflies — ModernGL"
    resource_dir = '.'
    gl_version = (3, 3)
    window_size = (1200, 800)
    aspect_ratio = None
    vsync = True
    resizable = True
    samples = 0

    # управление
    orbit_yaw = 0.0
    orbit_pitch = 0.25
    orbit_radius = 10.0
    mouse_down = False
    last_mouse = (0,0)
    keys_down = set()

    player_idx: Optional[int] = 0
    hud_on = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.ONE, mgl.ONE)     # аддитив для сцены (glow/трейл)
        self.ctx.enable(mgl.PROGRAM_POINT_SIZE)
        self.ctx.disable(mgl.DEPTH_TEST)

        # ---------- ШЕЙДЕРЫ ----------
        # Полилинии (каждый сегмент — одна входная точка!)
        self.poly_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                // Один "вершинный" элемент = один сегмент линии
                uniform mat4 u_mvp;
                in vec3 in_p0;
                in vec3 in_p1;
                in vec3 in_c0;
                in vec3 in_c1;
                in float in_w;
                out VS_OUT {
                    vec4 c0; vec4 c1;
                    vec3 col0; vec3 col1;
                    float w;
                } v;
                void main() {
                    v.c0   = u_mvp * vec4(in_p0,1.0);
                    v.c1   = u_mvp * vec4(in_p1,1.0);
                    v.col0 = in_c0;
                    v.col1 = in_c1;
                    v.w    = in_w;
                }
            ''',
            geometry_shader='''
                #version 330
                layout(points) in;                      // <— важно!
                layout(triangle_strip, max_vertices=4) out;

                uniform vec2 u_viewport;

                in VS_OUT {
                    vec4 c0; vec4 c1;
                    vec3 col0; vec3 col1;
                    float w;
                } v[];

                out vec2 g_uv;
                out vec3 g_col;

                void main() {
                    vec4 c0 = v[0].c0;
                    vec4 c1 = v[0].c1;

                    // в NDC
                    vec2 p0 = c0.xy / c0.w;
                    vec2 p1 = c1.xy / c1.w;

                    vec2 dir = normalize(p1 - p0);
                    vec2 nrm = vec2(-dir.y, dir.x);

                    float px = v[0].w;
                    vec2 ndc_off0 = nrm * px / u_viewport * 2.0;
                    vec2 ndc_off1 = ndc_off0;

                    vec4 off0 = vec4(ndc_off0 * c0.w, 0.0, 0.0);
                    vec4 off1 = vec4(ndc_off1 * c1.w, 0.0, 0.0);

                    gl_Position = c0 + off0; g_uv = vec2(-1.0, -1.0); g_col = v[0].col0; EmitVertex();
                    gl_Position = c0 - off0; g_uv = vec2( 1.0, -1.0); g_col = v[0].col0; EmitVertex();
                    gl_Position = c1 + off1; g_uv = vec2(-1.0,  1.0); g_col = v[0].col1; EmitVertex();
                    gl_Position = c1 - off1; g_uv = vec2( 1.0,  1.0); g_col = v[0].col1; EmitVertex();
                    EndPrimitive();
                }
            ''',
            fragment_shader='''
                #version 330
                in vec2 g_uv;
                in vec3 g_col;
                out vec4 f_color;
                void main() {
                    float t = 1.0 - abs(g_uv.x);
                    float alpha = smoothstep(0.0, 0.9, t);
                    vec3 col = g_col * (0.8 + 0.2*t);
                    col += 0.25 * (1.0 - t) * g_col;   // мягкое свечение по краям
                    f_color = vec4(col, alpha);
                }
            '''
        )

        # Точечные спрайты (ядро звезды)
        self.point_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 u_mvp;
                in vec3 in_pos;
                in vec3 in_col;
                in float in_size_px;
                out vec3 v_col;
                void main() {
                    vec4 c = u_mvp * vec4(in_pos,1.0);
                    gl_Position = c;
                    gl_PointSize = in_size_px;
                    v_col = in_col;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_col;
                out vec4 f_color;
                void main() {
                    vec2 uv = gl_PointCoord * 2.0 - 1.0;
                    float r = length(uv);
                    float a = smoothstep(1.0, 0.0, r);
                    vec3 col = v_col * (0.7 + 0.3*(1.0 - r));
                    f_color = vec4(col, a);
                }
            '''
        )

        # Комбинатор (accum)
        self.combine_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos; out vec2 v_uv;
                void main(){ v_uv=(in_pos+1.0)*0.5; gl_Position=vec4(in_pos,0,1); }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D u_prev, u_scene;
                uniform float u_decay;
                in vec2 v_uv; out vec4 f_color;
                void main(){
                    vec3 prev = texture(u_prev, v_uv).rgb;
                    vec3 sc   = texture(u_scene, v_uv).rgb;
                    f_color = vec4(prev*u_decay + sc, 1.0);
                }
            '''
        )

        # Финальный экран: туманность + HUD
        self.copy_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos; out vec2 v_uv;
                void main(){ v_uv=(in_pos+1.0)*0.5; gl_Position=vec4(in_pos,0,1); }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D u_tex;
                uniform float u_time, u_nebula_strength, u_hud_life;
                uniform vec2  u_res;
                in vec2 v_uv; out vec4 f_color;

                float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
                float noise(vec2 p){
                    vec2 i = floor(p), f = fract(p);
                    float a = hash(i), b = hash(i+vec2(1,0));
                    float c = hash(i+vec2(0,1)), d = hash(i+vec2(1,1));
                    vec2 u = f*f*(3.0-2.0*f);
                    return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
                }
                float fbm(vec2 p){
                    float v=0.0,a=0.5;
                    for(int i=0;i<5;i++){ v+=a*noise(p); p*=2.0; a*=0.55; }
                    return v;
                }

                float rect(vec2 uv, vec2 pos, vec2 size, float r){
                    vec2 d = abs(uv-pos)-size; vec2 q = max(d,0.0);
                    return 1.0 - clamp(length(q)-r, 0.0, 1.0);
                }

                void main(){
                    vec2 res = u_res;
                    vec3 acc = texture(u_tex, v_uv).rgb;

                    // туманность
                    vec2 n_uv = (v_uv*vec2(res.x/res.y,1.0))*0.85
                                + vec2(0.23*u_time*0.03, 0.17*u_time*0.02);
                    float n = fbm(n_uv*2.0);
                    float n2 = fbm(n_uv*1.3 + 7.0);
                    vec3 neb = vec3(0.15,0.10,0.25)*n + vec3(0.02,0.06,0.08)*n2;
                    neb *= 0.35 + 0.65*smoothstep(0.0,1.0,n);

                    vec3 outc = acc + u_nebula_strength * neb;

                    if (u_hud_life >= 0.0) {
                        vec2 uv = v_uv;
                        float frame = rect(uv, vec2(0.12, 0.08), vec2(0.10, 0.015), 0.006);
                        float fillw = 0.10 * clamp(u_hud_life, 0.0, 1.0);
                        float fill = rect(uv, vec2(0.12 - (0.10-fillw)*0.5, 0.08), vec2(fillw*0.5, 0.010), 0.004);
                        vec3 bar = mix(vec3(0.05), vec3(0.25,0.85,0.55), fill);
                        outc = mix(outc, outc*0.6 + bar, 0.75*frame + 0.85*fill);
                    }

                    outc = pow(outc, vec3(1.0/2.2));
                    f_color = vec4(outc, 1.0);
                }
            '''
        )

        # полноэкранный квад
        quad = np.array([[-1,-1],[1,-1],[-1,1], [1,-1],[1,1],[-1,1]], np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.quad_vao_combine = self.ctx.vertex_array(self.combine_prog, [(self.quad_vbo, '2f', 'in_pos')])
        self.quad_vao_copy    = self.ctx.vertex_array(self.copy_prog,    [(self.quad_vbo, '2f', 'in_pos')])

        self._build_targets()
        for fbo in self.accum_fbo:
            fbo.use(); self.ctx.clear(0,0,0,1)

        # симуляция
        self.butterflies: List[Butterfly] = [Butterfly.make() for _ in range(INIT_BUTTERFLIES)]
        self.stars: List[Star] = [Star.make() for _ in range(10)]
        self.star_spawn_acc = 0.0
        self.void_birth_acc = 0.0

        # буферы для полилиний (по 1 записи на сегмент)
        self.seg_p0 = self.ctx.buffer(reserve=4)
        self.seg_p1 = self.ctx.buffer(reserve=4)
        self.seg_c0 = self.ctx.buffer(reserve=4)
        self.seg_c1 = self.ctx.buffer(reserve=4)
        self.seg_w  = self.ctx.buffer(reserve=4)
        self.poly_vao = self.ctx.vertex_array(
            self.poly_prog,
            [
                (self.seg_p0, '3f', 'in_p0'),
                (self.seg_p1, '3f', 'in_p1'),
                (self.seg_c0, '3f', 'in_c0'),
                (self.seg_c1, '3f', 'in_c1'),
                (self.seg_w,  '1f', 'in_w')
            ]
        )

        # буферы для точек (ядра звёзд)
        self.pt_pos = self.ctx.buffer(reserve=4)
        self.pt_col = self.ctx.buffer(reserve=4)
        self.pt_siz = self.ctx.buffer(reserve=4)
        self.point_vao = self.ctx.vertex_array(
            self.point_prog,
            [
                (self.pt_pos, '3f', 'in_pos'),
                (self.pt_col, '3f', 'in_col'),
                (self.pt_siz, '1f', 'in_size_px'),
            ]
        )

        self.last_time = time.perf_counter()

    # ---------- targets ----------
    def _build_targets(self):
        w, h = self.wnd.size
        w = max(1, w); h = max(1, h)
        self.scene_tex = self.ctx.texture((w, h), 4, dtype='f2')
        self.scene_fbo = self.ctx.framebuffer(self.scene_tex)
        self.scene_fbo.viewport = (0, 0, w, h)

        self.accum_tex = [self.ctx.texture((w, h), 4, dtype='f2'),
                          self.ctx.texture((w, h), 4, dtype='f2')]
        for t in self.accum_tex: t.filter = (mgl.LINEAR, mgl.LINEAR)
        self.accum_fbo = [self.ctx.framebuffer(self.accum_tex[0]),
                          self.ctx.framebuffer(self.accum_tex[1])]
        self.accum_fbo[0].viewport = (0, 0, w, h)
        self.accum_fbo[1].viewport = (0, 0, w, h)
        self.accum_idx = 0

    # ---------- ввод ----------
    def on_resize(self, w, h):
        self._build_targets()
        for fbo in self.accum_fbo: fbo.use(); self.ctx.clear(0,0,0,1)

    def on_mouse_press(self, x, y, button):
        if button == 1:
            self.mouse_down = True
            self.last_mouse = (x, y)

    def on_mouse_release(self, x, y, button):
        if button == 1:
            self.mouse_down = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.mouse_down:
            self.orbit_yaw   += dx * 0.005
            self.orbit_pitch += dy * 0.005
            self.orbit_pitch = max(-1.2, min(1.2, self.orbit_pitch))

    def on_mouse_scroll(self, x, y, dx, dy):
        self.orbit_radius *= 0.9 ** dy
        self.orbit_radius = min(25.0, max(4.0, self.orbit_radius))

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys
        if action == k.ACTION_PRESS:
            self.keys_down.add(key)
            if key == k.SPACE:
                for fbo in self.accum_fbo: fbo.use(); self.ctx.clear(0,0,0,1)
            elif key == k.TAB:
                if self.butterflies:
                    if self.player_idx is None: self.player_idx = 0
                    else: self.player_idx = (self.player_idx + 1) % len(self.butterflies)
            elif key == k.H:
                self.hud_on = not self.hud_on
            elif key == k.MINUS:
                self.change_decay(-0.01)
            elif key == k.EQUAL:
                self.change_decay(+0.01)
            elif key == k.NUMBER_1:
                self.change_nebula(-0.02)
            elif key == k.NUMBER_2:
                self.change_nebula(+0.02)
        elif action == k.ACTION_RELEASE:
            if key in self.keys_down: self.keys_down.remove(key)

    def change_decay(self, d):
        global ACCUM_DECAY
        ACCUM_DECAY = float(np.clip(ACCUM_DECAY + d, 0.80, 0.98))
        print("ACCUM_DECAY =", ACCUM_DECAY)

    def change_nebula(self, d):
        global NEBULA_STRENGTH
        NEBULA_STRENGTH = float(np.clip(NEBULA_STRENGTH + d, 0.0, 0.5))
        print("NEBULA_STRENGTH =", NEBULA_STRENGTH)

    # ---------- камера ----------
    def _build_mvp(self, frame_time: float) -> Matrix44:
        yaw, pitch, r = self.orbit_yaw, self.orbit_pitch, self.orbit_radius
        eye = Vector3([math.cos(pitch)*math.sin(yaw)*r,
                       math.sin(pitch)*r,
                       math.cos(pitch)*math.cos(yaw)*r])
        view = Matrix44.look_at(eye, (0,0,0), (0,1,0), dtype=np.float32)
        w, h = self.wnd.size
        proj = Matrix44.perspective_projection(FOV, max(1e-6, w/float(h)), 0.1, 100.0, dtype=np.float32)
        return proj * view

    # ---------- симуляция ----------
    def _spawn_butterfly_near(self, parent: Butterfly):
        if len(self.butterflies) >= MAX_BUTTERFLIES: return
        b = Butterfly.make(copy=parent)
        b.center4[:3] += rng.normal(0, 0.05, 3).astype(np.float32)
        b.center4[3]  += rng.normal(0, 0.03)
        b.fade_in = 0.0
        self.butterflies.append(b)

    def _spawn_butterfly_void(self):
        if len(self.butterflies) >= MAX_BUTTERFLIES: return
        c4 = rng.uniform(-1.2,1.2,4).astype(np.float32)
        b = Butterfly.make(center4=c4)
        b.life = 0.25; b.fade_in = 0.0
        self.butterflies.append(b)

    def _player_input(self, b: Butterfly, dt: float):
        k = self.wnd.keys
        if not self.keys_down: return
        dir3 = np.zeros(3, np.float32)
        if k.W in self.keys_down: dir3[2] -= 1
        if k.S in self.keys_down: dir3[2] += 1
        if k.A in self.keys_down: dir3[0] -= 1
        if k.D in self.keys_down: dir3[0] += 1
        if k.Q in self.keys_down: dir3[1] -= 1
        if k.E in self.keys_down: dir3[1] += 1
        if np.linalg.norm(dir3) > 0:
            dir3 = dir3 / (np.linalg.norm(dir3)+1e-6)
            boost = 2.0 if (k.LEFT_SHIFT in self.keys_down or k.RIGHT_SHIFT in self.keys_down) else 1.0
            b.acc4[:3] += 2.0 * boost * dir3

    def _update_sim(self, dt: float, tnow: float):
        # звезды
        self.star_spawn_acc += STAR_SPAWN_RATE*dt
        while self.star_spawn_acc >= 1.0 and len(self.stars) < STAR_MAX:
            self.stars.append(Star.make()); self.star_spawn_acc -= 1.0
        for s in self.stars: s.update(dt)
        self.stars = [s for s in self.stars if s.alive]

        # бабочки
        N = len(self.butterflies)
        centers3 = np.array([b.center4[:3] for b in self.butterflies], np.float32) if N>0 else np.zeros((0,3),np.float32)
        vels3    = np.array([b.vel4[:3] for b in self.butterflies],    np.float32) if N>0 else np.zeros((0,3),np.float32)

        for i, b in enumerate(self.butterflies):
            b.center4[3] += 0.18*math.sin(W_OSC*tnow)*dt
            b.mode_ttl -= dt
            if b.mode_ttl <= 0:
                b.mode = 'wander' if b.mode == 'seek' else 'seek'
                b.mode_ttl = rng.uniform(3.0, 6.0)
                if b.mode == 'wander':
                    b.wander_center = b.center4[:3] + rng.normal(0,0.6,3).astype(np.float32)

            if b.mode == 'seek' and (b.target < 0 or b.target >= len(self.stars)) and self.stars:
                d = [np.linalg.norm(b.center4[:3]-s.pos3) for s in self.stars]
                b.target = int(np.argmin(d))

            b.acc4 *= 0.0
            b.acc4 += rng.normal(0.0, OU_SIGMA, 4).astype(np.float32) * math.sqrt(max(dt,1e-3))
            b.acc4 += -OU_BETA*b.vel4

            b.flap_phase += b.flap_omega * dt
            flap = 0.5*(1.0 - math.cos(b.flap_phase))
            v3 = b.vel4[:3]; vnorm = np.linalg.norm(v3) + 1e-6
            heading = v3 / vnorm
            b.acc4[:3] += FLAP_PUSH * flap * heading
            b.acc4[2]  += LIFT_GAIN  * flap

            if self.player_idx is not None and i == self.player_idx:
                self._player_input(b, dt)

            if b.mode == 'seek' and 0 <= b.target < len(self.stars):
                dir3 = self.stars[b.target].pos3 - b.center4[:3]
                b.acc4[:3] += SEEK_GAIN * dir3
            else:
                want = (b.wander_center - b.center4[:3])
                b.acc4[:3] += WANDER_GAIN * want

            if N > 1:
                diffs = centers3 - centers3[i]
                d2 = np.einsum('ij,ij->i', diffs, diffs)
                mask = (d2 > 1e-9) & (d2 < NEIGHBOR_R**2)
                neighbors = np.where(mask)[0]
                if neighbors.size > 0:
                    sep = -np.sum(diffs[neighbors]/(d2[neighbors][:,None]+1e-6), axis=0)
                    ali = np.mean(vels3[neighbors], axis=0) - v3
                    coh = (np.mean(centers3[neighbors], axis=0) - centers3[i])
                    b.acc4[:3] += SEPARATION_W*sep + ALIGNMENT_W*ali + COHESION_W*coh

            aN = np.linalg.norm(b.acc4)
            if aN > ACCEL_LIMIT: b.acc4 *= ACCEL_LIMIT/(aN+1e-6)
            b.vel4 += b.acc4 * dt
            vN = np.linalg.norm(b.vel4)
            if vN > SPEED_LIMIT: b.vel4 *= (SPEED_LIMIT/(vN+1e-6))
            if b.mode != 'seek': b.vel4 *= (1.0 - 0.20*dt)
            b.center4 += b.vel4 * dt
            b.angle1  += 0.6*dt; b.angle2 += 0.45*dt

            b.life = max(0.0, b.life - LIFE_DECAY_S*dt)
            b.fade_in = min(1.0, b.fade_in + FADEIN_SPEED*dt)

        # съели звезду?
        for b in self.butterflies:
            if 0 <= b.target < len(self.stars):
                if np.linalg.norm(b.center4[:3]-self.stars[b.target].pos3) <= STAR_RADIUS:
                    self.stars[b.target].alive = False
                    b.life = min(1.0, b.life + LIFE_FROM_STAR)
                    b.repro_cd = max(1.0, b.repro_cd)

        # спаривание
        births = []
        for i in range(len(self.butterflies)):
            for j in range(i+1, len(self.butterflies)):
                bi, bj = self.butterflies[i], self.butterflies[j]
                if bi.life >= REPRO_REQ and bj.life >= REPRO_REQ:
                    if np.linalg.norm(bi.center4[:3]-bj.center4[:3]) <= COLLISION_R:
                        births.append(i)
        for idx in births:
            if len(self.butterflies) < MAX_BUTTERFLIES:
                self._spawn_butterfly_near(self.butterflies[idx])
                self.butterflies[idx].life = max(0.0, self.butterflies[idx].life - REPRO_COST)

        # редкие спонтанные
        self.void_birth_acc += VOID_BIRTH_RATE*dt
        if self.void_birth_acc >= 1.0:
            self._spawn_butterfly_void(); self.void_birth_acc = 0.0

        # удалить погасших
        self.butterflies = [b for b in self.butterflies if b.alive()]
        if self.player_idx is not None and self.player_idx >= len(self.butterflies):
            self.player_idx = len(self.butterflies)-1 if self.butterflies else None

    # ---------- сбор геометрии ----------
    def _gather_scene(self, tnow: float):
        # Полилинии бабочек (по сегменту)
        seg_p0=[]; seg_p1=[]; seg_c0=[]; seg_c1=[]; seg_w=[]
        # Точки (ядра звёзд)
        pt_pos=[]; pt_col=[]; pt_size=[]
        # Шипы‑звёзды (как линии, пойдут в общий массив сегментов)
        spike_p0=[]; spike_p1=[]; spike_c0=[]; spike_c1=[]; spike_w=[]

        for i, b in enumerate(self.butterflies):
            flap = 0.5*(1.0 - math.cos(b.flap_phase))
            P = (BASE_SHAPE_4D.copy()).astype(np.float32)
            P[:,0] *= (1.0 + WING_AMP*(flap - 0.5))
            P[:,1] *= (1.0 - WING_AMP*(flap - 0.5))
            P[:,3] *= (1.0 + 0.22*(flap - 0.5))
            rotate_plane(P,0,3,b.angle1); rotate_plane(P,1,2,b.angle2)
            P += b.center4
            P3 = project_4d_to_3d(P, d=3.5)

            w_now = P[:,3]
            w_norm = (w_now - w_now.min()) / max(1e-6, (w_now.max()-w_now.min()))
            hue = (w_norm + b.hue0 + 0.08*math.sin(0.35*tnow)) % 1.0
            sat = np.full_like(hue, b.sat, np.float32)
            val = np.full_like(hue, b.val, np.float32)
            rgb = hsv_to_rgb_np(hue, sat, val)
            alpha_factor = max(MIN_VIS_ALPHA, 0.15 + 0.85*b.life*b.fade_in)
            rgb *= alpha_factor
            if self.player_idx is not None and i == self.player_idx:
                rgb = np.clip(rgb * 1.3, 0, 1)

            P3d = P3[::DOWNSAMPLE_STEP]
            C3d = rgb[::DOWNSAMPLE_STEP]
            if P3d.shape[0] >= 2:
                p0 = P3d[:-1]; p1 = P3d[1:]
                c0 = C3d[:-1]; c1 = C3d[1:]
                wpx = LINE_W_BFLY[0] + (LINE_W_BFLY[1]-LINE_W_BFLY[0]) * float(np.clip(b.life*b.fade_in, 0, 1))
                seg_p0.append(p0); seg_p1.append(p1)
                seg_c0.append(c0); seg_c1.append(c1)
                seg_w.append(np.full((p0.shape[0],1), wpx, np.float32))

        for i, s in enumerate(self.stars):
            base_col = np.array([1.0, 0.92, 0.62], np.float32)
            tw = 0.75 + 0.25*math.sin(2.0*tnow + i*0.7)
            col = base_col * tw
            pt_pos.append(s.pos3); pt_col.append(col); pt_size.append([STAR_BASE_PX])

            r_out = STAR_BASE_PX * 0.040 * (1.0 + 0.20*math.sin(2.2*tnow + i*0.5))
            r_in  = r_out * SPIKE_INNER_FR
            for d in s.dirs:
                p0 = s.pos3 + d * r_in; p1 = s.pos3 + d * r_out
                spike_p0.append(p0); spike_p1.append(p1)
                spike_c0.append(col); spike_c1.append(col)
                spike_w.append(LINE_W_SPIKE)

        if seg_p0:
            seg_p0 = np.vstack(seg_p0).astype(np.float32)
            seg_p1 = np.vstack(seg_p1).astype(np.float32)
            seg_c0 = np.vstack(seg_c0).astype(np.float32)
            seg_c1 = np.vstack(seg_c1).astype(np.float32)
            seg_w  = np.vstack(seg_w).astype(np.float32)
        else:
            seg_p0 = np.zeros((0,3),np.float32); seg_p1 = np.zeros((0,3),np.float32)
            seg_c0 = np.zeros((0,3),np.float32); seg_c1 = np.zeros((0,3),np.float32)
            seg_w  = np.zeros((0,1),np.float32)

        if spike_p0:
            sp0 = np.array(spike_p0, np.float32); sp1 = np.array(spike_p1, np.float32)
            sc0 = np.array(spike_c0, np.float32); sc1 = np.array(spike_c1, np.float32)
            sw  = np.array(spike_w,  np.float32).reshape(-1,1)
            seg_p0 = np.vstack([seg_p0, sp0]); seg_p1 = np.vstack([seg_p1, sp1])
            seg_c0 = np.vstack([seg_c0, sc0]); seg_c1 = np.vstack([seg_c1, sc1])
            seg_w  = np.vstack([seg_w,  sw])

        pt_pos = np.array(pt_pos, np.float32) if pt_pos else np.zeros((0,3),np.float32)
        pt_col = np.array(pt_col, np.float32) if pt_col else np.zeros((0,3),np.float32)
        pt_size= np.array(pt_size,np.float32) if pt_size else np.zeros((0,1),np.float32)

        return seg_p0, seg_p1, seg_c0, seg_c1, seg_w, pt_pos, pt_col, pt_size

    # ---------- рендер ----------
    def on_render(self, t, frame_time): self.render(t, frame_time)

    def render(self, time_sec: float, frame_time: float):
        now = time.perf_counter()
        dt = min(0.05, max(1e-4, now - self.last_time))
        self.last_time = now

        self._update_sim(dt, now)
        seg_p0, seg_p1, seg_c0, seg_c1, seg_w, pt_pos, pt_col, pt_size = self._gather_scene(now)

        # Pass 1: сцена -> scene_fbo
        self.scene_fbo.use()
        w, h = self.wnd.size
        self.scene_fbo.viewport = (0, 0, w, h)
        self.ctx.clear(0,0,0,0)
        self.ctx.blend_func = (mgl.ONE, mgl.ONE)

        mvp = self._build_mvp(frame_time)

        # полилинии (как POINTS: 1 запись = 1 сегмент)
        if seg_p0.shape[0] > 0:
            self.poly_prog['u_mvp'].write(mvp.astype('f4').tobytes())
            self.poly_prog['u_viewport'].value = (float(w), float(h))
            self.seg_p0.orphan(len(seg_p0)*12); self.seg_p0.write(seg_p0.tobytes())
            self.seg_p1.orphan(len(seg_p1)*12); self.seg_p1.write(seg_p1.tobytes())
            self.seg_c0.orphan(len(seg_c0)*12); self.seg_c0.write(seg_c0.tobytes())
            self.seg_c1.orphan(len(seg_c1)*12); self.seg_c1.write(seg_c1.tobytes())
            self.seg_w.orphan(len(seg_w)*4);    self.seg_w.write(seg_w.tobytes())
            self.poly_vao.render(mode=mgl.POINTS)

        # ядра звёзд
        if pt_pos.shape[0] > 0:
            self.point_prog['u_mvp'].write(mvp.astype('f4').tobytes())
            self.pt_pos.orphan(len(pt_pos)*12); self.pt_pos.write(pt_pos.tobytes())
            self.pt_col.orphan(len(pt_col)*12); self.pt_col.write(pt_col.tobytes())
            self.pt_siz.orphan(len(pt_size)*4); self.pt_siz.write(pt_size.tobytes())
            self.point_vao.render(mode=mgl.POINTS)

        # Pass 2: combine(prev, scene) -> accum
        self.ctx.blend_func = (mgl.ONE, mgl.ZERO)
        prev_idx = self.accum_idx; next_idx = 1 - self.accum_idx
        self.accum_fbo[next_idx].use()
        self.accum_fbo[next_idx].viewport = (0, 0, w, h)
        self.combine_prog['u_prev'].value = 0
        self.combine_prog['u_scene'].value = 1
        self.combine_prog['u_decay'].value = ACCUM_DECAY
        self.accum_tex[prev_idx].use(location=0)
        self.scene_tex.use(location=1)
        self.quad_vao_combine.render()
        self.accum_idx = next_idx

        # Pass 3: экран + туманность + HUD
        self.ctx.screen.use()
        self.copy_prog['u_tex'].value = 0
        self.copy_prog['u_time'].value = now
        self.copy_prog['u_res'].value = (float(w), float(h))
        life = self.butterflies[self.player_idx].life if (self.player_idx is not None and self.butterflies) else 0.0
        self.copy_prog['u_hud_life'].value = life if self.hud_on else -1.0
        self.copy_prog['u_nebula_strength'].value = NEBULA_STRENGTH
        self.accum_tex[self.accum_idx].use(location=0)
        self.quad_vao_copy.render()

if __name__ == '__main__':
    mglw.run_window_config(ButterflyApp)
