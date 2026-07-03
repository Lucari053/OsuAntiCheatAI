import math
import random
import os
import bisect
from dataclasses import dataclass
from osrparse import Replay, GameMode
from osrparse import Key
from pebble import ProcessPool
from concurrent.futures import TimeoutError as FutureTimeoutError
from tqdm import tqdm
import argparse

from parser.edit_osr import patch_osr, get_beatmap_replay_hash, set_replay_hash
from parser.beatmap_parser import BeatmapParser
from dataset.prepaire_beatmap import build_beatmap_objects
from dataset.mods import apply_mods_to_difficulty
from dataset.prepaire_replay import validate_replay
from dataset.utils import get_beatmap_hash



@dataclass
class AimAssistParam:
    strength:      float = 0.8
    fov:           float = 120.0
    assist_window: float = 70.0
    max_speed:     float = 40.0
    humanization:  float = 0.0
    seed:          int   = None

    def random(self):
        self.strength      = random.uniform(0.15, 1.0)
        self.fov           = random.uniform(40.0, 200.0)
        self.assist_window = random.uniform(25.0, 110.0)
        self.max_speed     = random.uniform(15.0, 60.0)
        self.humanization  = random.uniform(0.0, 0.8)
        self.seed          = random.randint(0, 2**31 - 1)
        

@dataclass
class AimCorrectionParam:
    strength:         float = 0.9
    window:           int   = 3
    tolerance_factor: float = 1.0
    landing_factor:   float = 0.4
    flow:             float = 0.2
    seed:             int   = None

    def random(self):
        self.strength         = random.uniform(0.4, 1.0)
        self.window           = random.randint(1, 6)
        self.tolerance_factor = random.uniform(0.5, 1.2)
        self.landing_factor   = random.uniform(0.2, 0.7)
        self.flow             = random.uniform(0.0, 0.35)
        self.seed             = random.randint(0, 2**31 - 1)

@dataclass
class RelaxParam:
    offset_ms: float = 0.0
    jitter_ms: float = 2.0
    hold_ms:   float = 30.0
    alternate: bool = True
    seed:      int = None

    def random(self):
        self.offset_ms = random.uniform(-25.0, 25.0)
        self.jitter_ms = random.uniform(0.0, 14.0)
        self.hold_ms   = random.uniform(18.0, 55.0)
        self.alternate = random.random() < 0.9
        self.seed      = random.randint(0, 2**31 - 1)

@dataclass
class HumanAutobotParam:
    spread:         float = 5.0
    curve_strength: float = 1.0
    spin_radius:    float = 60.0
    spin_speed:     float = 0.05
    seed:           int   = None

    def random(self):
        self.spread         = random.uniform(0.0, 12.0)
        self.curve_strength = random.uniform(0.0, 1.2)
        self.spin_radius    = random.uniform(40.0, 90.0)
        self.spin_speed     = random.uniform(0.03, 0.08)
        self.seed           = random.randint(0, 2**31 - 1)

@dataclass
class TapAssistParam:
    strength:             float = 0.6
    jitter_ms:            float = 3.0
    rescue_chance:        float = 0.0
    rescue_radius_factor: float = 1.5
    hold_ms:              float = 30.0
    seed:                 int   = None

    def random(self):
        self.strength             = random.uniform(0.3, 1.0)
        self.jitter_ms            = random.uniform(0.0, 12.0)
        self.rescue_chance        = random.uniform(0.0, 0.6)
        self.rescue_radius_factor = random.uniform(1.0, 2.0)
        self.hold_ms              = random.uniform(18.0, 40.0)
        self.seed                 = random.randint(0, 2**31 - 1)


def _validate_bd(bd, max_coord: float = 1e4, max_t: float = 120_000) -> bool:
    for o in bd:
        if not (math.isfinite(o.x) and math.isfinite(o.y) and math.isfinite(o.t)):
            return False
        if abs(o.x) > max_coord or abs(o.y) > max_coord:
            return False
        if o.t < 0 or o.t > max_t:
            return False
    return True

class CheatSynth:
    """
    Synth cheat replays by taking a reference replay
    """
    osr_path:    str = ""
    map_content: str = ""
    output_path: str = ""

    aim_assist_param:     AimAssistParam     = None
    aim_correction_param: AimCorrectionParam = None
    relax_param:          RelaxParam         = None
    human_autobot_param:  HumanAutobotParam  = None
    tap_assist_param:     TapAssistParam     = None
    
    def init_replay(
        self,
        osr_path: str,
        map_content: str,
        output_path: str = "",
    ):
        """
        Initialize reference replay and associated beatmap 
        to proceed next synth_cheat().

        Args:
            osr_path (str):    reference replay path
            map_content (str): utf-8 string of the beatmap file OR beatmap path
            output_path (str): the output path, if it is not a file, auto add hash to filename
        """
        self.osr_path    = osr_path
        self.output_path = output_path

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        

        extension = (os.path.splitext(map_content)[1].lower() 
                     if os.path.exists(map_content) else "")
            
        # .osu file or content process parse
        if extension in (".osu", ""):
            
            # Get beatmap content
            if extension == ".osu":
                with open(map_content, 'r', encoding='utf-8') as f:
                    map_content = f.read()
            
            # Parse
            parser = BeatmapParser()
            for line in map_content.split("\n"):
                parser.read_line(line)
            
            beatmap = parser.build_beatmap()
            self.beatmap = {
                "objects": build_beatmap_objects(beatmap),
                "cs":      float(beatmap.get("CircleSize", 4)),
                "od":      float(beatmap.get("OverallDifficulty", 8))
            }
        

    def init_parameter(
        self,
        aim_assist_param:     AimAssistParam     = None,
        aim_correction_param: AimCorrectionParam = None,
        relax_param:          RelaxParam         = None,
        human_autobot_param:  HumanAutobotParam  = None,
        tap_assist_param:     TapAssistParam     = None
    ):
        """
        Initialize cheat parameter manually,
        *None* = non used
        \n**Warning**, cheats type are not all compatible and are auto skip
        """
        if aim_assist_param is not None:
            self.aim_assist_param = aim_assist_param
        if aim_correction_param is not None:
            self.aim_correction_param = aim_correction_param
        if relax_param is not None:
            self.relax_param = relax_param
        if human_autobot_param is not None:
            self.human_autobot_param = human_autobot_param
        if tap_assist_param is not None:
            self.tap_assist_param = tap_assist_param


    def random_param(self):
        """
        Initialize randomly all parameters
        """
        self.aim_assist_param = None
        self.aim_correction_param = None
        self.relax_param = None
        self.human_autobot_param = None

        # Compatibility
        profile = random.choice([
            "aim_assist", "aim_correction", "relax", "tap_assist",
            "aim_assist+relax", "aim_correction+relax",
            "aim_assist+tap_assist", "aim_correction+tap_assist",
            "aim_assist+aim_correction", "aim_assist+aim_correction+relax",
            "aim_assist+aim_correction+tap_assist",
            "autobot+relax", "autobot+tap_assist",
        ])
        if "autobot" in profile:
            self.human_autobot_param = HumanAutobotParam(); self.human_autobot_param.random()
        if "aim_assist" in profile:
            self.aim_assist_param = AimAssistParam(); self.aim_assist_param.random()
        if "aim_correction" in profile:
            self.aim_correction_param = AimCorrectionParam(); self.aim_correction_param.random()
        if "relax" in profile:
            self.relax_param = RelaxParam(); self.relax_param.random()
        if "tap_assist" in profile:
            self.tap_assist_param = TapAssistParam(); self.tap_assist_param.random()

    def aim_assist(self, raw, bd):
        param = self.aim_assist_param
        rng = random.Random(param.seed)

        def transform(frames):
            abs_real, acc = [], 0.0
            for o in bd:
                acc += o.t
                abs_real.append(acc)

            W = param.assist_window  # fenêtre d'objets "actifs" autour de maintenant (ms)
            replay_time = 0.0
            
            for i, frame in enumerate(frames):
                t, x, y, k = frame
                if t == -12345:
                    continue
                if i < 2 and x == 256.0 and y == -500.0:
                    continue
                
                replay_time += t

                # Objets actifs = temps de hit dans [now - W, now + W]
                lo = bisect.bisect_left(abs_real, replay_time - W)
                hi = bisect.bisect_right(abs_real, replay_time + W)

                # Le plus proche SPATIALEMENT du curseur (hors spinner)
                best_d, target = None, None
                for oi in range(lo, hi):
                    o = bd[oi]
                    if o.object_type == 4:
                        continue
                    d = math.hypot(x - o.x, y - o.y)
                    if best_d is None or d < best_d:
                        best_d, target = d, o
                
                # Si aucune cible ou cible hors du champ de vision (FOV), on ignore
                if target is None or param.fov <= 0 or best_d > param.fov:
                    continue

                # 1. Lissage de l'activation (quadratique pour éviter un "snap" rigide sur les bords du FOV)
                activation = (1.0 - (best_d / param.fov)) ** 2
                
                # 2. Indépendance au framerate (delta time 't')
                # On normalise par rapport à une frame standard d'environ 10ms
                dt_norm = max(1.0, t) / 10.0
                
                # La force dépend maintenant du paramètre de base, de la proximité, et du temps écoulé
                pull = min(1.0, param.strength * activation * 0.15 * dt_norm)

                dx = (target.x - x) * pull
                dy = (target.y - y) * pull
                disp = math.hypot(dx, dy)
                
                # 3. Vitesse maximale normalisée par le temps
                current_max_speed = param.max_speed * dt_norm
                if disp > current_max_speed and disp > 0.0:
                    s = current_max_speed / disp
                    dx *= s
                    dy *= s

                # 4. Bruit adaptatif (Humanisation)
                if param.humanization > 0:
                    # Le bruit est proportionnel à la force d'attraction pour masquer l'assistance
                    noise = param.humanization * 2.5 * pull
                    dx += rng.gauss(0, noise)
                    dy += rng.gauss(0, noise)

                # Application des modifications au curseur
                frame[1] = round(x + dx, 3)
                frame[2] = round(y + dy, 3)

            return frames
    
        new_raw = patch_osr(raw, transform)
        return new_raw

    def aim_correction(self, raw, bd):
        param = self.aim_correction_param
        tolerance = param.tolerance_factor * self.hit_radius
        landing = param.landing_factor * self.hit_radius

        def transform(frames):
            # 1. Calcul des temps absolus (Map)
            abs_times = [0.0] * len(bd)
            acc = 0.0
            for i, obj in enumerate(bd):
                acc += obj.t
                abs_times[i] = acc

            clickable_data = [] 
            for idx, (t_abs, obj) in enumerate(zip(abs_times, bd)):
                if obj.object_type in (0, 1): 
                    clickable_data.append((idx, t_abs, obj))
            
            if not clickable_data:
                return frames
            
            clickable_times = [item[1] for item in clickable_data]

            # 2. Extraction stricte des clics du joueur (Replay)
            replay_time, prev_pressed = 0.0, False
            clicks = []
            
            for i, frame in enumerate(frames):
                t, x, y, k = frame
                if t == -12345:
                    continue
                
                # IMPORTANT: Toujours accumuler le temps pour rester synchro
                replay_time += t
                
                if i < 2 and x == 256.0 and y == -500.0:
                    continue
                    
                pressed = bool(k & (Key.K1 | Key.K2 | Key.M1 | Key.M2))
                if pressed and not prev_pressed:
                    clicks.append((i, replay_time, x, y))
                prev_pressed = pressed

            n_frames = len(frames)
            offsets_x = [0.0] * n_frames
            offsets_y = [0.0] * n_frames
            
            consumed_indices = set()
            anchors = [] # Liste des keyframes: (frame_index, offset_x, offset_y)

            # 3. Calcul de la correction REQUISE à l'instant exact du clic
            for click_i, click_time, cx, cy in clicks:
                idx = bisect.bisect_left(clickable_times, click_time)
                
                best_obj, best_dist = None, float('inf')
                best_global_idx, min_score = -1, float('inf')
                
                for cand_idx in range(max(0, idx - 4), min(len(clickable_data), idx + 4)):
                    global_idx, obj_t, obj = clickable_data[cand_idx]
                    if global_idx in consumed_indices: continue 
                        
                    dt = abs(obj_t - click_time)
                    if dt > self.hit_window: continue
                        
                    dist = math.hypot(cx - obj.x, cy - obj.y)
                    score = dt + (dist * 1.5)
                    
                    if score < min_score:
                        min_score, best_obj, best_dist, best_global_idx = score, obj, dist, global_idx
                
                if best_obj is None or best_dist > self.hit_radius * 4:
                    continue
                    
                consumed_indices.add(best_global_idx)
                
                # Si déjà parfait, offset de zéro (mais on l'ajoute comme ancre pour protéger le stream)
                if best_dist <= tolerance:
                    anchors.append((click_i, 0.0, 0.0))
                    continue

                ratio = max(0.0, 1.0 - (landing / best_dist))
                pull = ratio * param.strength
                anchors.append((click_i, (best_obj.x - cx) * pull, (best_obj.y - cy) * pull))

            # 4. Interpolation NON-ADDITIVE des offsets (LE correctif)
            for idx in range(len(anchors)):
                pos, ox, oy = anchors[idx]
                
                left_bound = pos - param.window
                if idx > 0:
                    prev_pos = anchors[idx-1][0]
                    # Si les clics sont proches, on les relie directement sans revenir à zéro
                    if pos - prev_pos <= 2 * param.window:
                        left_bound = prev_pos
                
                right_bound = pos + param.window
                if idx < len(anchors) - 1:
                    next_pos = anchors[idx+1][0]
                    if next_pos - pos <= 2 * param.window:
                        right_bound = next_pos
                
                # Lissage côté gauche (Fade In ou Interpolation avec le clic précédent)
                if left_bound == pos - param.window:
                    for i in range(left_bound, pos):
                        if i < 0: continue
                        u = (pos - i) / param.window
                        w = 0.5 * (1 + math.cos(math.pi * u))
                        offsets_x[i] = ox * w; offsets_y[i] = oy * w
                else:
                    prev_pos, prev_ox, prev_oy = anchors[idx-1]
                    for i in range(prev_pos + 1, pos):
                        if i < 0: continue
                        u = (i - prev_pos) / (pos - prev_pos)
                        w = 0.5 * (1 - math.cos(math.pi * u))
                        offsets_x[i] = prev_ox * (1 - w) + ox * w
                        offsets_y[i] = prev_oy * (1 - w) + oy * w
                        
                # Fixer le point d'impact exact
                if 0 <= pos < n_frames:
                    offsets_x[pos] = ox; offsets_y[pos] = oy
                    
                # Lissage côté droit (Fade Out seulement s'il n'y a pas d'autre clic direct après)
                if right_bound == pos + param.window:
                    for i in range(pos + 1, right_bound + 1):
                        if i >= n_frames: continue
                        u = (i - pos) / param.window
                        w = 0.5 * (1 + math.cos(math.pi * u))
                        offsets_x[i] = ox * w; offsets_y[i] = oy * w

            # 5. Application Finale
            flow = param.flow
            prev_ox, prev_oy = 0.0, 0.0
            
            for i, frame in enumerate(frames):
                t, x, y, k = frame
                if t == -12345 or (i < 2 and x == 256.0 and y == -500.0):
                    continue
                
                ox, oy = offsets_x[i], offsets_y[i]
                smooth_ox = ox * (1.0 - flow) + prev_ox * flow
                smooth_oy = oy * (1.0 - flow) + prev_oy * flow
                
                frame[1] = round(x + smooth_ox, 3)
                frame[2] = round(y + smooth_oy, 3)
                
                prev_ox, prev_oy = smooth_ox, smooth_oy

            return frames

        return patch_osr(raw, transform)            

    def relax(self, raw, bd):
        param = self.relax_param
        rng = random.Random(param.seed)
        K1, K2 = 4, 8
        CLICK_MASK = 1 | 2 | 4 | 8

        # 1. Calcul des temps absolus (Map)
        # FIX DT: On retire la division par self.rate ! Le temps est natif.
        abs_t = [0.0] * len(bd)
        acc = 0.0
        for i, obj in enumerate(bd):
            acc += obj.t
            abs_t[i] = acc

        # 2. Générer les intervalles de clics désirés (Brut)
        raw_presses = []
        i = 0
        while i < len(bd):
            ot = bd[i].object_type
            if ot == 0: # Circle
                head = abs_t[i]
                raw_presses.append((head, head + param.hold_ms))
                i += 1
            elif ot == 1: # Slider
                head = abs_t[i]
                j = i + 1
                while j < len(bd) and bd[j].object_type in (2, 3):
                    j += 1
                tail = abs_t[j - 1]
                raw_presses.append((head, max(tail, head + param.hold_ms)))
                i = j
            elif ot == 4: # Spinner
                start = abs_t[i]
                end = abs_t[i + 1] if i + 1 < len(bd) and bd[i + 1].object_type == 4 else start + param.hold_ms
                raw_presses.append((start, end))
                i += 2
            else:
                i += 1

        # 3. Nettoyer les intervalles (Appliquer Offset/Jitter)
        presses = []
        for n, (t0, t1) in enumerate(raw_presses):
            start_t = t0 + param.offset_ms + rng.gauss(0, param.jitter_ms)
            end_t = t1 + param.offset_ms
            
            if end_t <= start_t:
                end_t = start_t + 1.0

            key = K1 if (not param.alternate or n % 2 == 0) else K2
            presses.append([start_t, end_t, key])

        # FIX OVERLAP : Si un clic déborde sur le suivant, on le coupe net !
        # Cela force un "front montant" (relâchement puis pression) indispensable pour valider un Hit.
        for n in range(len(presses) - 1):
            current_start, current_end, current_key = presses[n]
            next_start, next_end, next_key = presses[n + 1]
            
            if current_end >= next_start - 2.0:
                presses[n][1] = next_start - 2.0

        def transform(frames):
            valid = []
            vtimes = []
            acc = 0.0
            
            # Extraction des temps du replay
            for idx, fr in enumerate(frames):
                if fr[0] == -12345:
                    continue
                acc += fr[0]
                valid.append(idx)
                vtimes.append(acc)
            
            # Effacer tous les clics originaux du joueur
            for idx in valid:
                frames[idx][3] &= ~CLICK_MASK

            # Ré-injection chirurgicale des clics
            for start_t, end_t, key in presses:
                # bisect_left donne la PREMIÈRE frame qui survient au moment exact du start_t ou juste après.
                # Fini les clics en avance à cause d'un mauvais arrondi vers la frame précédente.
                v0 = bisect.bisect_left(vtimes, start_t)
                hi = bisect.bisect_right(vtimes, end_t)
                
                # FIX VOID CLICKS : Si le clic est si court qu'aucune frame ne tombe dedans, 
                # on force l'écriture sur au moins 1 frame pour que le jeu l'enregistre.
                if v0 == hi and v0 < len(vtimes):
                    hi += 1
                    
                for v in range(v0, hi):
                    if v < len(valid):
                        frames[valid[v]][3] |= key

            return frames
    
        return patch_osr(raw, transform)

    def human_autobot(self, raw, bd):
        param = self.human_autobot_param
        rng = random.Random(param.seed)

        # 1. Calcul des temps (FIX DT : Pas de division par self.rate)
        abs_t = [0.0] * len(bd)
        acc = 0.0
        for i, obj in enumerate(bd):
            acc += obj.t
            abs_t[i] = acc

        # 2. Génération intelligente des Waypoints
        wp_t, wp_x, wp_y = [], [], []
        prev_t = -1.0
        
        for idx, obj in enumerate(bd):
            if obj.object_type == 4:
                continue  # Les spinners sont gérés à part

            t_abs = abs_t[idx]
            
            # FIX ALLER-RETOUR : Si deux objets ont le même timestamp (stack parfait),
            # on décale artificiellement de 1ms pour empêcher l'interpolation de reculer.
            if t_abs <= prev_t:
                t_abs = prev_t + 1.0

            dt = t_abs - prev_t if prev_t >= 0 else 1000.0
            
            # FIX STREAMS : Réduit drastiquement l'erreur aléatoire (spread) 
            # sur les objets rapprochés pour tracer des lignes droites.
            current_spread = param.spread
            if dt < 120:
                current_spread *= max(0.1, dt / 120.0)
                
            # Les ticks internes d'un slider (2) et les fins de sliders (3) 
            # ne doivent presque pas avoir de spread pour éviter que le curseur ne tremble.
            if obj.object_type in (2, 3):
                current_spread *= 0.2

            ox = obj.x + rng.gauss(0, current_spread)
            oy = obj.y + rng.gauss(0, current_spread)
            
            wp_t.append(t_abs)
            wp_x.append(ox)
            wp_y.append(oy)
            
            prev_t = t_abs

        # 3. Spinners (Inchangé)
        spinners, k = [], 0
        while k < len(bd):
            if bd[k].object_type == 4:
                start = abs_t[k]
                end = abs_t[k + 1] if k + 1 < len(bd) and bd[k + 1].object_type == 4 else start
                spinners.append((start, end, bd[k].x, bd[k].y))
                k += 2
            else:
                k += 1
        
        # 4. Interpolation avec ANTI-OVERSHOOT
        def hermite(p0, p1, p2, p3, u, cs):
            m1 = cs * (p2 - p0) * 0.5
            m2 = cs * (p3 - p1) * 0.5
            
            # FIX FLICKS EXAGÉRÉS : Clamp des tangentes.
            # L'élan mathématique (tangente) ne peut pas excéder 1.5x la distance réelle.
            # Cela empêche les courbes de former des "noeuds papillons" entre deux objets proches.
            dist = abs(p2 - p1)
            max_tan = dist * 1.5
            if abs(m1) > max_tan: m1 = math.copysign(max_tan, m1)
            if abs(m2) > max_tan: m2 = math.copysign(max_tan, m2)

            u2, u3 = u * u, u * u * u
            return ((2*u3 - 3*u2 + 1) * p1 + (u3 - 2*u2 + u) * m1
                     + (-2*u3 + 3*u2) * p2 + (u3 - u2) * m2)
        
        def transform(frames):
            ft, acc = [], 0.0
            for fr in frames:
                if fr[0] == -12345:
                    ft.append(acc)
                    continue
                acc += fr[0]
                ft.append(acc)
            
            n = len(wp_t)
            for idx, fr in enumerate(frames):
                if fr[0] == -12345 or (idx < 2 and fr[1] == 256.0 and fr[2] == -500.0):
                    continue
                
                T = ft[idx]

                # Rendu Spinner
                spin = next(((s0, cx, cy) for (s0, s1, cx, cy) in spinners if s0 <= T <= s1), None)
                if spin is not None:
                    s0, cx, cy = spin
                    ang = (T - s0) * param.spin_speed
                    fr[1] = round(cx + math.cos(ang) * param.spin_radius, 3)
                    fr[2] = round(cy + math.sin(ang) * param.spin_radius, 3)
                    continue
                
                # Rendu standard
                if n == 0: 
                    continue
                if T <= wp_t[0]:
                    fr[1], fr[2] = round(wp_x[0], 3), round(wp_y[0], 3)
                    continue
                if T >= wp_t[-1]:
                    fr[1], fr[2] = round(wp_x[-1], 3), round(wp_y[-1], 3)
                    continue

                j = bisect.bisect_right(wp_t, T) - 1
                t0, t1 = wp_t[j], wp_t[j + 1]
                
                # Sécurité anti-division par zéro
                u = (T - t0) / (t1 - t0) if t1 > t0 else 0.0

                i0, i3 = max(0, j - 1), min(n - 1, j + 2)
                fr[1] = round(hermite(wp_x[i0], wp_x[j], wp_x[j+1], wp_x[i3], u, param.curve_strength), 3)
                fr[2] = round(hermite(wp_y[i0], wp_y[j], wp_y[j+1], wp_y[i3], u, param.curve_strength), 3)

            return frames
    
        return patch_osr(raw, transform)

    def tap_assist(self, raw, bd):
        param = self.tap_assist_param
        rng = random.Random(param.seed)
        PRESS_MASK = 1 | 2 | 4 | 8

        abs_t, acc = [], 0.0
        for obj in bd:
            acc += obj.t
            abs_t.append(acc)

        clickable_times, clickable_objs = [], []
        for idx, obj in enumerate(bd):
            if obj.object_type in (0, 1):
                clickable_times.append(abs_t[idx])
                clickable_objs.append(obj)

        def transform(frames):
            valid, vtimes, acc = [], [], 0.0
            for idx, fr in enumerate(frames):
                if fr[0] == -12345:
                    continue
                acc += fr[0]
                valid.append(idx)
                vtimes.append(acc)

            def cursor_at(t):
                v = min(range(len(vtimes)), key=lambda i: abs(vtimes[i] - t))
                fr = frames[valid[v]]
                return fr[1], fr[2]

            presses = []
            press_start, press_bits = None, 0
            for v, idx in enumerate(valid):
                bits = frames[idx][3] & PRESS_MASK
                if bits and press_start is None:
                    press_start, press_bits = vtimes[v], bits
                elif not bits and press_start is not None:
                    presses.append((press_start, vtimes[v], press_bits))
                    press_start, press_bits = None, 0
            if press_start is not None:
                presses.append((press_start, vtimes[-1], press_bits))

            matched = set()
            shifts = []
            for start, end, bits in presses:
                j = bisect.bisect_left(clickable_times, start)
                best = None
                for cand in (j - 1, j):
                    if 0 <= cand < len(clickable_times):
                        dt = abs(clickable_times[cand] - start)
                        if best is None or dt < best[0]:
                            best = (dt, cand)
                if best is None or best[0] > self.hit_window:
                    continue

                matched.add(best[1])
                ideal_time = clickable_times[best[1]]
                hold = end - start
                new_start = start + (ideal_time - start) * param.strength + rng.gauss(0, param.jitter_ms)
                shifts.append((start, end, new_start, new_start + hold, bits))

            # Secours sur quasi-raté : clickable jamais matché, curseur proche au bon moment
            for ci, (t_ideal, target) in enumerate(zip(clickable_times, clickable_objs)):
                if ci in matched:
                    continue
                if rng.random() >= param.rescue_chance:
                    continue

                cx, cy = cursor_at(t_ideal)
                if math.hypot(cx - target.x, cy - target.y) > self.hit_radius * param.rescue_radius_factor:
                    continue  # trop loin, même un vrai cheat ne cliquerait pas n'importe où

                new_start = t_ideal + rng.gauss(0, param.jitter_ms)
                shifts.append((None, None, new_start, new_start + param.hold_ms, Key.K1.value))

            # Efface les appuis déplacés (les secours n'ont pas d'ancien intervalle)
            for old_start, old_end, _, _, bits in shifts:
                if old_start is None:
                    continue
                lo = bisect.bisect_left(vtimes, old_start)
                hi = bisect.bisect_right(vtimes, old_end)
                for v in range(lo, hi):
                    frames[valid[v]][3] &= ~bits

            shifts.sort(key=lambda s: s[2])  # par new_start
            shifts.sort(key=lambda s: s[2])
            for _, _, new_start, new_end, bits in shifts:
                # front montant = frame la PLUS PROCHE de new_start (supprime le biais retard)
                v0 = bisect.bisect_left(vtimes, new_start)
                if v0 > 0 and (v0 >= len(vtimes) or
                            abs(vtimes[v0 - 1] - new_start) < abs(vtimes[v0] - new_start)):
                    v0 -= 1
                hi = max(bisect.bisect_right(vtimes, new_end), v0 + 1)

                # front montant garanti (anti-fusion burst)
                if v0 - 1 >= 0:
                    frames[valid[v0 - 1]][3] &= ~bits
                for v in range(v0, hi):
                    frames[valid[v]][3] |= bits

                for v in range(lo, hi):
                    frames[valid[v]][3] |= bits

            return frames

        return patch_osr(raw, transform)


    def synth_cheat(self, skip_invalid=False):
        """
        Synth a cheats play with reference
        """

        # Get replay data
        replay = Replay.from_path(self.osr_path)
        if replay.mode != GameMode.STD:  # Only support osu std
            return False
        
        with open(self.osr_path, 'rb') as f:
            raw = f.read()
        
        bd = self.beatmap['objects']
        
        
        # Check if play is valid
        if len(bd) <= 0 or not _validate_bd(bd):
            return False
        if skip_invalid:
            if not validate_replay(replay, bd):
                return False # Invalid play
        
        # Get difficulty and apply mods
        cs_raw = float(self.beatmap['cs'])
        od_raw = float(self.beatmap['od'])
        cs_adj, od_adj = apply_mods_to_difficulty(cs_raw, od_raw, replay.mods)
        self.hit_radius = 54.4 - 4.48 * cs_adj
        self.hit_window = 200.0

        # Apply cheats
        if self.human_autobot_param is not None:
            raw = self.human_autobot(raw, bd)
        else:
            if self.aim_assist_param is not None:
                raw = self.aim_assist(raw, bd)
            if self.aim_correction_param is not None:
                raw = self.aim_correction(raw, bd)
        
        if self.relax_param is not None:
            raw = self.relax(raw, bd)
        # Disable for now, because not perform well, need improvment
        #elif self.tap_assist_param is not None:
        #    raw = self.tap_assist(raw, bd)

        # Generate unique hash for the modified replay
        new_hash = get_beatmap_hash(raw)
        raw = set_replay_hash(raw, new_hash)

        if os.path.isdir(self.output_path):
            output_path = os.path.join(self.output_path, f"{new_hash}.osr")
        else:
            output_path = self.output_path
        
        with open(output_path, 'wb') as f:
            f.write(raw)
        
        return True


def process_cheat_file(osr_path: str, beatmap_path: str, output_folder: str):
    
    synth = CheatSynth()
    synth.init_replay(osr_path, beatmap_path, output_folder)
    
    synth.random_param()
    synth.synth_cheat(skip_invalid=True)

def process_cheat_folder(
    replay_folder:  str, 
    beatmap_folder: str, 
    output_folder:  str, 
    max_workers:    int   = None,
    timeout:        float = 2.0,
    limit:          int   = 100,
):

    args = []
    print("Load dataset...")

    os.makedirs(output_folder, exist_ok=True)
    if limit is None: limit = len(os.listdir(replay_folder))
    
    pbar = tqdm(total=min(limit, len(os.listdir(replay_folder))))
    for filename in os.listdir(replay_folder)[200_000:]:

        replay_path = os.path.join(replay_folder, filename)

        with open(replay_path, 'rb') as f:
            replay_content = f.read()
        map_hash, _ = get_beatmap_replay_hash(replay_content)

        beatmap_path = os.path.join(beatmap_folder, f"{map_hash}.osu")
            
        if not os.path.exists(beatmap_path):
            continue

        args.append((replay_path, beatmap_path, output_folder))
        pbar.update(1)

        if pbar.n >= limit: break
    
    print("Process dataset...")
    with ProcessPool(max_workers=max_workers or os.cpu_count()) as pool:
        futures = [pool.schedule(process_cheat_file, args=a, timeout=timeout) for a in args]
        for fut in tqdm(futures, total=len(futures)):
            try:
                fut.result()
            except FutureTimeoutError:
                pass
            except Exception:
                pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--replays-folder",  help="Folder with replay (.osr) to take as reference",   required=True, type=str)
    parser.add_argument("--beatmaps-folder", help="Folder with beatmaps '{beatmap_hash}.osu'",        required=True, type=str)
    parser.add_argument("--output-folder",   help="The output folder for cheat .osr replay generate", required=True, type=str)
    parser.add_argument("--max-workers", default=None, required=False, type=int)
    parser.add_argument("--limit", help="Max cheat to generate", default=None, required=False, type=int)

    args = parser.parse_args()

    process_cheat_folder(
        args.replays_folder, 
        args.beatmaps_folder, 
        args.output_folder,
        max_workers=args.max_workers,
        timeout=2,
        limit=args.limit
    )

    #process_cheat_file(
    #    osr_path,
    #    beatmap_path,
    #    "temp/cheat.osr"
    #)
