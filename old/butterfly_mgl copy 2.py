# -*- coding: utf-8 -*-
# butterfly_mgl.py — 4D-бабочки на ModernGL: плавные взмахи, шипастые звёзды,
# glow+трейлы через накопление кадров и процедурная туманность.
#
# pip install moderngl moderngl-window pyrr numpy

import math, time
from dataclasses import dataclass, field
from typing import List
import numpy as np
import moderngl as mgl
import moderngl_window as mglw
from pyrr import Matrix44, Vector3

# ------------------------ ПАРАМЕТРЫ ------------------------
MAX_BUTTERFLIES      = 16
INIT_BUTTERFLIES     = 6
POINTS_PER_BUTTERFLY = 1400
WING_AMP             = 0.45
FLAP_FREQ_RANGE      = (0.28, 0.38)  # Гц — медленные взмахи

SPEED_LIMIT   = 0.7
ACCEL_LIMIT   = 1.0
OU_BETA       = 0.9
OU_SIGMA      = 0.55
SEEK_GAIN     = 1.0
WANDER_GAIN   = 0.30
LIFT_GAIN     = 0.28
FLAP_PUSH     = 0.70
W_OSC         = 0.8

# Боид‑правила
NEIGHBOR_R      = 0.9
SEPARATION_W    = 0.9
ALIGNMENT_W     = 0.25
COHESION_W      = 0.30

# Жизнь/размножение
SPLIT_BOOST     = 0.40
SPLIT_PROB      = 0.55
STAR_RADIUS     = 0.20
FADE_RATE       = 0.010
BASE_FADE       = 0.008
VOID_BIRTH_RATE = 0.010
COLLISION_R     = 0.20
COLLISION_BIRTH_PROB = 0.010

# Звёзды
STAR_MAX        = 28
STAR_SPAWN_RATE = 0.55
STAR_TTL        = (7.0, 13.0)
STAR_SPIKES_MIN = 10
STAR_SPIKES_MAX = 18
SPIKE_INNER_FR  = 0.35
STAR_BASE_PX    = 6.0   # ядро звезды — вдвое меньше прежнего

# Акумуляция / glow
ACCUM_DECAY     = 0.90   # длина хвоста
FOV = 55.0
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

def butterfly_curve_4d(n=POINTS_PER_BUTTERFLY, scale=0.20):
    t = np.linspace(0, 24*np.pi, n).astype(np.float32)
    r = np.exp(np.cos(t)) - 2*np.cos(4*t) + np.sin(t/12)**5
    x = r*np.sin(t); y = r*np.cos(t)
    z = 0.55*np.sin(3*t); w = 0.55*np.cos(2*t)
    P4 = np.stack([x, y, z, w], axis=1).astype(np.float32)
    P4 /= (np.max(np.linalg.norm(P4[:, :3], axis=1)) + 1e-6)
    P4 *= scale
    return P4, t

BASE_SHAPE_4D, _ = butterfly_curve_4d()

# ------------------------ СИМУЛЯЦИЯ -----------------------
@dataclass
class Star:
    pos3: np.ndarray
    ttl: float
    born: float
    dirs: np.ndarray  # (k,3) нормализованные направления шипов
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
        return Star(np.array([x,y,z], np.float32), float(rng.uniform(*STAR_TTL)), time.perf_counter(), u)
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
    vital: float = 0.6
    fade_in: float = 0.0
    target: int = -1
    flap_phase: float = field(default_factory=lambda: rng.uniform(0, 2*np.pi))
    flap_omega: float = field(default_factory=lambda: 2*np.pi*rng.uniform(*FLAP_FREQ_RANGE))
    hue0: float = field(default_factory=lambda: rng.uniform(0,1))
    sat:  float = field(default_factory=lambda: rng.uniform(0.7, 0.95))
    val:  float = field(default_factory=lambda: rng.uniform(0.80, 0.98))
    mode: str = field(default_factory=lambda: 'seek' if rng.random() < 0.6 else 'wander')
    mode_ttl: float = field(default_factory=lambda: rng.uniform(3.0, 6.0))
    wander_center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    wander_omega: np.ndarray = field(default_factory=lambda: rng.uniform(0.12, 0.28, 3).astype(np.float32))
    repro_cd: float = field(default_factory=lambda: rng.uniform(2.0, 4.0))
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
                          s, vital=min(1.0, copy.vital*0.75+0.2))
            b.wander_center = b.center4[:3].copy()
            return b
        c = rng.uniform(-1.0,1.0,4).astype(np.float32) if center4 is None else center4.astype(np.float32)
        v = rng.normal(0, 0.25, 4).astype(np.float32)
        a = np.zeros(4, np.float32)
        b = Butterfly(c, v, a, rng.uniform(0,2*np.pi), rng.uniform(0,2*np.pi), 1.0)
        b.wander_center = b.center4[:3].copy()
        return b
    def alive(self): return self.vital > 0.02

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

# ------------------------ ОКНО / РЕНДЕР --------------------
class ButterflyApp(mglw.WindowConfig):
    title = "4D Butterflies — ModernGL"
    resource_dir = '.'
    gl_version = (3, 3)
    window_size = (1200, 800)
    aspect_ratio = None
    samples = 0
    vsync = True
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.scene_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 u_mvp;
                uniform float u_px_scale;
                uniform vec3  u_light_dir;  // "освещение"
                in vec3 in_pos;
                in vec3 in_col;
                in float in_size;
                out vec3 v_col;
                out float v_size;
                out float v_depth;
                out vec3 v_ldir;
                void main() {
                    vec4 p = u_mvp * vec4(in_pos, 1.0);
                    gl_Position = p;
                    float ps = max(1.0, in_size * u_px_scale / max(0.0001, p.w));
                    gl_PointSize = ps;
                    v_col = in_col;
                    v_size = ps;
                    v_depth = p.z/p.w;
                    v_ldir = normalize(u_light_dir);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_col;
                in float v_size;
                in float v_depth;
                in vec3 v_ldir;
                out vec4 f_color;
                void main() {
                    vec2 uv = gl_PointCoord * 2.0 - 1.0;
                    float r = length(uv);
                    // имитируем нормаль "сферической" частицы для освещения
                    float z = sqrt(max(0.0, 1.0 - r*r));
                    vec3 nrm = normalize(vec3(uv, z));
                    float lambert = clamp(dot(nrm, normalize(v_ldir)) * 0.8 + 0.2, 0.0, 1.2);
                    float alpha = smoothstep(1.0, 0.0, r);
                    vec3 col = v_col * lambert;
                    float glow = smoothstep(1.2, 0.0, r);
                    col += 0.45 * glow * v_col;
                    f_color = vec4(col, alpha);
                }
            '''
        )
        self.lines_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 u_mvp;
                in vec3 in_pos;
                in vec4 in_col;
                out vec4 v_col;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos,1.0);
                    v_col = in_col;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec4 v_col;
                out vec4 f_color;
                void main() { f_color = v_col; }
            '''
        )
        self.combine_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos;
                out vec2 v_uv;
                void main(){
                    v_uv = (in_pos + 1.0)*0.5;
                    gl_Position = vec4(in_pos,0.0,1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D u_prev;
                uniform sampler2D u_scene;
                uniform float u_decay;
                uniform float u_time;
                uniform vec2  u_res;

                in vec2 v_uv; out vec4 f_color;

                float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
                float noise(vec2 p){
                    vec2 i = floor(p); vec2 f = fract(p);
                    float a = hash(i);
                    float b = hash(i+vec2(1,0));
                    float c = hash(i+vec2(0,1));
                    float d = hash(i+vec2(1,1));
                    vec2 u = f*f*(3.0-2.0*f);
                    return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
                }
                float fbm(vec2 p){
                    float v=0.0,a=0.5;
                    for(int i=0;i<5;i++){ v+=a*noise(p); p*=2.0; a*=0.55; }
                    return v;
                }

                void main(){
                    vec2 uv = v_uv;

                    // лёгкий "блюр" сцены (супердешёвый glow)
                    vec3 sc = texture(u_scene, uv).rgb * 0.36;
                    sc += texture(u_scene, uv + vec2( 1.0/u_res.x, 0)).rgb * 0.08;
                    sc += texture(u_scene, uv + vec2(-1.0/u_res.x, 0)).rgb * 0.08;
                    sc += texture(u_scene, uv + vec2(0,  1.0/u_res.y)).rgb * 0.08;
                    sc += texture(u_scene, uv + vec2(0, -1.0/u_res.y)).rgb * 0.08;
                    sc += texture(u_scene, uv + vec2( 1.0/u_res.x,  1.0/u_res.y)).rgb * 0.05;
                    sc += texture(u_scene, uv + vec2(-1.0/u_res.x,  1.0/u_res.y)).rgb * 0.05;
                    sc += texture(u_scene, uv + vec2( 1.0/u_res.x, -1.0/u_res.y)).rgb * 0.05;
                    sc += texture(u_scene, uv + vec2(-1.0/u_res.x, -1.0/u_res.y)).rgb * 0.05;

                    vec3 prev = texture(u_prev, uv).rgb;

                    // туманность (FBM)
                    vec2 n_uv = (uv*vec2(u_res.x/u_res.y,1.0))*0.85 + vec2(0.23*u_time*0.03, 0.17*u_time*0.02);
                    float n = fbm(n_uv*2.0);
                    float n2 = fbm(n_uv*1.3 + 7.0);
                    vec3 neb = vec3(0.15,0.10,0.25)*n + vec3(0.02,0.06,0.08)*n2;
                    neb *= 0.35 + 0.65*smoothstep(0.0,1.0,n);

                    vec3 outc = prev*u_decay + sc + 0.35*neb;
                    outc = pow(outc, vec3(1.0/2.2));
                    f_color = vec4(outc, 1.0);
                }
            '''
        )
        self.copy_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos; out vec2 v_uv;
                void main(){ v_uv=(in_pos+1.0)*0.5; gl_Position=vec4(in_pos,0,1); }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D u_tex;
                in vec2 v_uv; out vec4 f_color;
                void main(){ f_color = texture(u_tex, v_uv); }
            '''
        )

        quad = np.array([[-1,-1],[1,-1],[-1,1], [1,-1],[1,1],[-1,1]], np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.quad_vao_combine = self.ctx.vertex_array(self.combine_prog, [(self.quad_vbo, '2f', 'in_pos')])
        self.quad_vao_copy    = self.ctx.vertex_array(self.copy_prog,    [(self.quad_vbo, '2f', 'in_pos')])

        self._build_targets()

        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.ONE, mgl.ONE)
        self.ctx.disable(mgl.DEPTH_TEST)

        self.butterflies: List[Butterfly] = [Butterfly.make() for _ in range(INIT_BUTTERFLIES)]
        self.stars: List[Star] = [Star.make() for _ in range(10)]
        self.void_birth_acc = 0.0
        self.star_spawn_acc = 0.0

        self.pts_vbo   = self.ctx.buffer(reserve=4)
        self.col_vbo   = self.ctx.buffer(reserve=4)
        self.size_vbo  = self.ctx.buffer(reserve=4)
        self.scene_vao = self.ctx.vertex_array(
            self.scene_prog,
            [(self.pts_vbo,  '3f', 'in_pos'),
             (self.col_vbo,  '3f', 'in_col'),
             (self.size_vbo, '1f', 'in_size')]
        )

        self.line_pos = self.ctx.buffer(reserve=4)
        self.line_col = self.ctx.buffer(reserve=4)
        self.lines_vao = self.ctx.vertex_array(
            self.lines_prog,
            [(self.line_pos, '3f', 'in_pos'),
             (self.line_col, '4f', 'in_col')]
        )

        self.last_time = time.perf_counter()
        self.angle = 0.0

    # ----- framebuffer targets -----
    def _build_targets(self):
        w, h = self.wnd.size
        if w <= 0 or h <= 0:
            w, h = 1, 1
        self.scene_tex = self.ctx.texture((w, h), 4, dtype='f2')
        self.scene_fbo = self.ctx.framebuffer(self.scene_tex)
        self.accum_tex = [self.ctx.texture((w, h), 4, dtype='f2'),
                          self.ctx.texture((w, h), 4, dtype='f2')]
        for t in self.accum_tex:
            t.filter = (mgl.LINEAR, mgl.LINEAR)
        self.accum_fbo = [self.ctx.framebuffer(self.accum_tex[0]),
                          self.ctx.framebuffer(self.accum_tex[1])]
        self.accum_idx = 0

    # NB: у твоей версии moderngl-window нужен on_resize/on_render:
    def on_resize(self, width: int, height: int):
        self.resize(width, height)
    def resize(self, width: int, height: int):
        self._build_targets()

    def on_key_event(self, key, action, modifiers):
        self.key_event(key, action, modifiers)
    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                for fbo in self.accum_fbo:
                    fbo.use(); self.ctx.clear(0,0,0,1)
            elif key == self.wnd.keys.N and len(self.butterflies) < MAX_BUTTERFLIES:
                self._spawn_butterfly_void()

    # ------------------- ЛОГИКА -------------------------
    def _spawn_butterfly_near(self, parent: Butterfly):
        if len(self.butterflies) >= MAX_BUTTERFLIES: return
        b = Butterfly.make(copy=parent)
        b.center4[:3] += rng.normal(0, 0.05, 3).astype(np.float32)
        b.center4[3]  += rng.normal(0, 0.03)
        self.butterflies.append(b)

    def _spawn_butterfly_void(self):
        if len(self.butterflies) >= MAX_BUTTERFLIES: return
        c4 = rng.uniform(-1.2,1.2,4).astype(np.float32)
        b = Butterfly.make(center4=c4)
        b.vital = 0.35; b.fade_in = 0.0
        self.butterflies.append(b)

    def _update_sim(self, dt: float, tnow: float):
        # звёзды
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
            v3 = b.vel4[:3]
            vnorm = np.linalg.norm(v3) + 1e-6
            heading = v3 / vnorm
            b.acc4[:3] += FLAP_PUSH * flap * heading
            b.acc4[2]  += LIFT_GAIN  * flap

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
            b.angle1  += 0.6*dt
            b.angle2  += 0.45*dt
            b.vital = max(0.0, b.vital - BASE_FADE*dt)
            b.fade_in = min(1.0, b.fade_in + 0.7*dt)

        newborns = []
        for b in self.butterflies:
            if 0 <= b.target < len(self.stars):
                if np.linalg.norm(b.center4[:3]-self.stars[b.target].pos3) <= STAR_RADIUS:
                    self.stars[b.target].alive = False
                    b.vital = min(1.0, b.vital + SPLIT_BOOST)
                    if b.repro_cd <= 0 and rng.random() < SPLIT_PROB:
                        newborns.append(b); b.repro_cd = rng.uniform(2.0, 4.0)
            if b.repro_cd > 0: b.repro_cd -= dt
        for p in newborns: self._spawn_butterfly_near(p)

        N = len(self.butterflies)
        if N >= 2 and N < MAX_BUTTERFLIES:
            centers3 = np.array([b.center4[:3] for b in self.butterflies], np.float32)
            for i in range(N):
                for j in range(i+1,N):
                    if np.linalg.norm(centers3[i]-centers3[j]) <= COLLISION_R:
                        if rng.uniform() < COLLISION_BIRTH_PROB*dt*60:
                            self._spawn_butterfly_near(self.butterflies[i])

        self.void_birth_acc += VOID_BIRTH_RATE*dt
        if self.void_birth_acc >= 1.0:
            self._spawn_butterfly_void(); self.void_birth_acc = 0.0
        self.butterflies = [b for b in self.butterflies if b.alive()]


    def _build_mvp(self, frame_time: float) -> Matrix44:
        w, h = self.wnd.size
        aspect = max(1e-6, w/float(h))
        proj = Matrix44.perspective_projection(FOV, aspect, 0.1, 100.0, dtype=np.float32)
        # раньше было self.wnd.time_delta — заменяем на frame_time
        dt = frame_time if (frame_time is not None and frame_time > 0.0) else 0.016
        self.angle += 0.12 * dt
        cam_pos = Vector3([math.sin(self.angle)*8.0, 3.0, 8.0*math.cos(self.angle)])
        view = Matrix44.look_at(cam_pos, (0.0,0.0,0.0), (0.0,1.0,0.0), dtype=np.float32)
        return proj * view

    def _gather_scene(self, tnow: float):
        pts=[]; cols=[]; sizes=[]
        for b in self.butterflies:
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
            alpha_factor = 0.30 + 0.70*b.vital*b.fade_in
            rgb *= alpha_factor
            pts.append(P3); cols.append(rgb); sizes.append(np.full((P3.shape[0],1), 1.8, np.float32))

        if pts:
            pts = np.vstack(pts).astype(np.float32)
            cols = np.vstack(cols).astype(np.float32)
            sizes = np.vstack(sizes).astype(np.float32)
        else:
            pts = np.zeros((1,3), np.float32); cols = np.zeros((1,3), np.float32); sizes = np.ones((1,1), np.float32)

        star_pos=[]; star_col=[]; star_size=[]; spike_pos=[]; spike_col=[]
        if self.stars:
            for i, s in enumerate(self.stars):
                star_pos.append(s.pos3)
                base_col = np.array([1.0, 0.92, 0.62], np.float32)
                tw = 0.75 + 0.25*math.sin(2.0*tnow + i*0.7)
                col = base_col * tw
                star_col.append(col); star_size.append([STAR_BASE_PX])
                r_out = STAR_BASE_PX * 0.018 * (1.0 + 0.15*math.sin(2.2*tnow + i*0.5))
                r_in  = r_out * SPIKE_INNER_FR
                for d in s.dirs:
                    p0 = s.pos3 + d * r_in; p1 = s.pos3 + d * r_out
                    spike_pos.append(p0); spike_pos.append(p1)
                    spike_col.append([col[0], col[1], col[2], 0.85]); spike_col.append([col[0], col[1], col[2], 0.85])

        if star_pos:
            star_pos = np.array(star_pos, np.float32); star_col = np.array(star_col, np.float32); star_size = np.array(star_size, np.float32)
            pts = np.vstack([pts, star_pos]); cols = np.vstack([cols, star_col]); sizes = np.vstack([sizes, star_size])
        spike_pos = np.array(spike_pos, np.float32) if spike_pos else np.zeros((0,3),np.float32)
        spike_col = np.array(spike_col, np.float32) if spike_col else np.zeros((0,4),np.float32)
        return pts, cols, sizes, spike_pos, spike_col

    # --- ВАЖНО: у твоей версии нужен именно on_render ---
    def on_render(self, time, frame_time):
        self.render(time, frame_time)

    def render(self, time_sec: float, frame_time: float):
        now = time.perf_counter()
        dt = min(0.05, max(1e-4, now - self.last_time))
        self.last_time = now
        self._update_sim(dt, now)
        pts, cols, sizes, lpos, lcol = self._gather_scene(now)

        # Pass 1: сцена в HDR‑текстуру
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.ONE, mgl.ONE)
        mvp = self._build_mvp(frame_time)
        self.scene_prog['u_mvp'].write(mvp.astype('f4').tobytes())
        self.scene_prog['u_px_scale'].value = 1.0
        self.scene_prog['u_light_dir'].value = (0.4, 0.7, 0.2)  # мягкий направленный свет

        self.pts_vbo.orphan(len(pts)*12);  self.pts_vbo.write(pts.tobytes())
        self.col_vbo.orphan(len(cols)*12); self.col_vbo.write(cols.tobytes())
        self.size_vbo.orphan(len(sizes)*4); self.size_vbo.write(sizes.tobytes())
        self.scene_vao.render(mode=mgl.POINTS)

        if len(lpos):
            self.lines_prog['u_mvp'].write(mvp.astype('f4').tobytes())
            self.line_pos.orphan(len(lpos)*12); self.line_pos.write(lpos.tobytes())
            self.line_col.orphan(len(lcol)*16); self.line_col.write(lcol.tobytes())
            self.lines_vao.render(mode=mgl.LINES)

        # Pass 2: combine(prev, scene) + туманность
        self.ctx.blend_func = (mgl.ONE, mgl.ZERO)
        w, h = self.wnd.size
        prev_idx = self.accum_idx; next_idx = 1 - self.accum_idx
        self.accum_fbo[next_idx].use()
        self.combine_prog['u_prev'].value = 0
        self.combine_prog['u_scene'].value = 1
        self.combine_prog['u_decay'].value = ACCUM_DECAY
        self.combine_prog['u_time'].value  = now
        self.combine_prog['u_res'].value   = (float(w), float(h))
        self.accum_tex[prev_idx].use(location=0)
        self.scene_tex.use(location=1)
        self.quad_vao_combine.render()
        self.accum_idx = next_idx

        # Pass 3: на экран
        self.ctx.screen.use()
        self.copy_prog['u_tex'].value = 0
        self.accum_tex[self.accum_idx].use(location=0)
        self.quad_vao_copy.render()

if __name__ == '__main__':
    mglw.run_window_config(ButterflyApp)
