import math
import random
import os
import bisect
from dataclasses import dataclass
from osrparse import Replay, GameMode
from osrparse import Key
from pebble import ProcessPool
from concurrent.futures import TimeoutError as FutureTimeoutError, as_completed
from tqdm import tqdm
import argparse
from enum import Enum

from parser.edit_osr import patch_osr, get_beatmap_replay_hash, set_replay_hash
from parser.beatmap_parser import BeatmapParser
from dataset.prepaire_beatmap import build_beatmap_objects
from dataset.mods import apply_mods_to_difficulty
from dataset.prepaire_replay import validate_replay
from dataset.utils import get_beatmap_hash

class CheatDifficulty(Enum):
    Easy = 0
    Hard = 1

@dataclass
class AimAssistParam:
    strength:      float = 0.8
    fov:           float = 120.0
    assist_window: float = 70.0
    max_speed:     float = 40.0
    humanization:  float = 0.0
    seed:          int   = None

    def random(self, difficulty: CheatDifficulty = None):
        if difficulty is None:
            self.strength      = random.uniform(0.15, 1.0)
            self.fov           = random.uniform(40.0, 200.0)
            self.assist_window = random.uniform(25.0, 110.0)
            self.max_speed     = random.uniform(15.0, 60.0)
            self.humanization  = random.uniform(0.0, 0.8)
        else:
            match difficulty:
                case CheatDifficulty.Easy:  # Blatant
                    self.strength      = random.uniform(0.6, 1.0)
                    self.fov           = random.uniform(120.0, 250.0)
                    self.assist_window = random.uniform(70.0, 120.0)
                    self.max_speed     = random.uniform(35.0, 70.0)
                    self.humanization  = random.uniform(0.0, 0.2)
                case CheatDifficulty.Hard:  # Closet
                    self.strength      = random.uniform(0.15, 0.4)
                    self.fov           = random.uniform(30.0, 70.0)
                    self.assist_window = random.uniform(20.0, 50.0)
                    self.max_speed     = random.uniform(10.0, 25.0)
                    self.humanization  = random.uniform(0.4, 0.85)
        
        self.seed = random.randint(0, 2**31 - 1)

@dataclass
class AimCorrectionParam:
    strength:         float = 0.9
    window:           int   = 3
    tolerance_factor: float = 1.0
    landing_factor:   float = 0.4
    flow:             float = 0.2
    seed:             int   = None

    def random(self, difficulty: CheatDifficulty = None):
        if difficulty is None:
            self.strength         = random.uniform(0.4, 1.0)
            self.window           = random.randint(1, 6)
            self.tolerance_factor = random.uniform(0.5, 1.2)
            self.landing_factor   = random.uniform(0.2, 0.7)
            self.flow             = random.uniform(0.0, 0.35)
        else:
            match difficulty:
                case CheatDifficulty.Easy:  # Blatant
                    self.strength         = random.uniform(0.8, 1.0)
                    self.window           = random.randint(1, 2)
                    self.tolerance_factor = random.uniform(0.2, 0.5)
                    self.landing_factor   = random.uniform(0.1, 0.3)
                    self.flow             = random.uniform(0.0, 0.1)
                case CheatDifficulty.Hard:  # Closet
                    self.strength         = random.uniform(0.3, 0.6)
                    self.window           = random.randint(4, 6)
                    self.tolerance_factor = random.uniform(0.8, 1.5)
                    self.landing_factor   = random.uniform(0.5, 0.8)
                    self.flow             = random.uniform(0.2, 0.45)
        
        self.seed = random.randint(0, 2**31 - 1)

@dataclass
class RelaxParam:
    offset_ms: float = 0.0
    jitter_ms: float = 2.0
    hold_ms:   float = 30.0
    alternate: bool = True
    seed:      int = None

    def random(self, difficulty: CheatDifficulty = None):
        if difficulty is None:
            self.offset_ms = random.uniform(-25.0, 25.0)
            self.jitter_ms = random.uniform(0.0, 14.0)
            self.hold_ms   = random.uniform(18.0, 55.0)
            self.alternate = random.random() < 0.9
        else:
            self.alternate = random.random() < 0.9
            match difficulty:
                case CheatDifficulty.Easy:  # Blatant
                    self.offset_ms = random.uniform(-5.0, 5.0)
                    self.jitter_ms = random.uniform(0.0, 1.5)
                    self.hold_ms   = random.uniform(25.0, 35.0)
                case CheatDifficulty.Hard:  # Closet
                    self.offset_ms = random.uniform(-20.0, 20.0)
                    self.jitter_ms = random.uniform(4.0, 12.0)
                    self.hold_ms   = random.uniform(15.0, 50.0)
        
        self.seed = random.randint(0, 2**31 - 1)

@dataclass
class HumanAutobotParam:
    spread:         float = 5.0
    curve_strength: float = 1.0
    spin_radius:    float = 60.0
    spin_speed:     float = 0.05
    seed:           int   = None

    def random(self, difficulty: CheatDifficulty = None):
        if difficulty is None:
            self.spread         = random.uniform(0.0, 12.0)
            self.curve_strength = random.uniform(0.0, 1.2)
            self.spin_radius    = random.uniform(40.0, 90.0)
            self.spin_speed     = random.uniform(0.03, 0.08)
        else:
            match difficulty:
                case CheatDifficulty.Easy:  # Blatant
                    self.spread         = random.uniform(0.0, 2.0)
                    self.curve_strength = random.uniform(0.8, 1.2)
                    self.spin_radius    = random.uniform(50.0, 70.0)
                    self.spin_speed     = random.uniform(0.05, 0.08)
                case CheatDifficulty.Hard:  # Closet
                    self.spread         = random.uniform(4.0, 10.0)
                    self.curve_strength = random.uniform(0.2, 0.6)
                    self.spin_radius    = random.uniform(30.0, 80.0)
                    self.spin_speed     = random.uniform(0.02, 0.05)
        
        self.seed = random.randint(0, 2**31 - 1)

@dataclass
class TapAssistParam:
    strength:             float = 0.6
    jitter_ms:            float = 3.0
    rescue_chance:        float = 0.0
    rescue_radius_factor: float = 1.5
    hold_ms:              float = 30.0
    seed:                 int   = None

    def random(self, difficulty: CheatDifficulty = None):
        if difficulty is None:
            self.strength             = random.uniform(0.3, 1.0)
            self.jitter_ms            = random.uniform(0.0, 12.0)
            self.rescue_chance        = random.uniform(0.0, 0.6)
            self.rescue_radius_factor = random.uniform(1.0, 2.0)
            self.hold_ms              = random.uniform(18.0, 40.0)
        else:
            match difficulty:
                case CheatDifficulty.Easy:  # Blatant
                    self.strength             = random.uniform(0.8, 1.0)
                    self.jitter_ms            = random.uniform(0.0, 2.0)
                    self.rescue_chance        = random.uniform(0.5, 0.8)
                    self.rescue_radius_factor = random.uniform(1.5, 2.2)
                    self.hold_ms              = random.uniform(25.0, 35.0)
                case CheatDifficulty.Hard:  # Closet
                    self.strength             = random.uniform(0.2, 0.5)
                    self.jitter_ms            = random.uniform(4.0, 10.0)
                    self.rescue_chance        = random.uniform(0.1, 0.3)
                    self.rescue_radius_factor = random.uniform(1.0, 1.4)
                    self.hold_ms              = random.uniform(15.0, 45.0)
        
        self.seed = random.randint(0, 2**31 - 1)


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


    def random_param(self, difficulty: CheatDifficulty = None):
        """
        Initialize randomly all parameters
        """
        self.aim_assist_param = None
        self.aim_correction_param = None
        self.relax_param = None
        self.human_autobot_param = None

        # Cheta Compatibility
        profile = random.choice([
            "aim_assist", "aim_correction", "relax", "tap_assist",
            "aim_assist+relax", "aim_correction+relax",
            "aim_assist+tap_assist", "aim_correction+tap_assist",
            "aim_assist+aim_correction", "aim_assist+aim_correction+relax",
            "aim_assist+aim_correction+tap_assist",
            "autobot+relax", "autobot+tap_assist",
        ])
        if "autobot" in profile:
            self.human_autobot_param = HumanAutobotParam(); self.human_autobot_param.random(difficulty)
        if "aim_assist" in profile:
            self.aim_assist_param = AimAssistParam(); self.aim_assist_param.random(difficulty)
        if "aim_correction" in profile:
            self.aim_correction_param = AimCorrectionParam(); self.aim_correction_param.random(difficulty)
        if "relax" in profile:
            self.relax_param = RelaxParam(); self.relax_param.random(difficulty)
        if "tap_assist" in profile:
            self.tap_assist_param = TapAssistParam(); self.tap_assist_param.random(difficulty)

    def aim_assist(self, raw, bd):
        param = self.aim_assist_param
        rng = random.Random(param.seed)

        def transform(frames):
            abs_real, acc = [], 0.0
            for o in bd:
                acc += o.t
                abs_real.append(acc)

            W = param.assist_window
            replay_time = 0.0
            
            for i, frame in enumerate(frames):
                t, x, y, k = frame
                if t == -12345:
                    continue
                if i < 2 and x == 256.0 and y == -500.0:
                    continue
                
                replay_time += t

                # Active objects = time hit on [now - W, now + W]
                lo = bisect.bisect_left(abs_real, replay_time - W)
                hi = bisect.bisect_right(abs_real, replay_time + W)

                # Nearest by cursor
                best_d, target = None, None
                for oi in range(lo, hi):
                    o = bd[oi]
                    if o.object_type == 4: # Do not process spinner
                        continue
                    d = math.hypot(x - o.x, y - o.y)
                    if best_d is None or d < best_d:
                        best_d, target = d, o
                

                if target is None or param.fov <= 0 or best_d > param.fov:
                    continue

                # Quadratic Smooth activation
                activation = (1.0 - (best_d / param.fov)) ** 2
                
                # Normalize frame rate
                dt_norm = max(1.0, t) / 10.0
                
                # Calcul force
                pull = min(1.0, param.strength * activation * 0.15 * dt_norm)

                dx = (target.x - x) * pull
                dy = (target.y - y) * pull
                disp = math.hypot(dx, dy)
                
                # Maximum speed normalize by time
                current_max_speed = param.max_speed * dt_norm
                if disp > current_max_speed and disp > 0.0:
                    s = current_max_speed / disp
                    dx *= s
                    dy *= s

                # Add noise (humanized)
                if param.humanization > 0:
                    # Noise = proportional to pull force
                    noise = param.humanization * 2.5 * pull
                    dx += rng.gauss(0, noise)
                    dy += rng.gauss(0, noise)

                # Apply force
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
            # Calcul absolute times
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

            # Extract player click to next time aligned with target
            replay_time, prev_pressed = 0.0, False
            clicks = []
            
            for i, frame in enumerate(frames):
                t, x, y, k = frame
                if t == -12345:
                    continue
                
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
            anchors = [] # (frame_index, offset_x, offset_y)

            # Calcul required correction when click
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
                
                # Already perfect
                if best_dist <= tolerance:
                    anchors.append((click_i, 0.0, 0.0))
                    continue

                ratio = max(0.0, 1.0 - (landing / best_dist))
                pull = ratio * param.strength
                anchors.append((click_i, (best_obj.x - cx) * pull, (best_obj.y - cy) * pull))

            # Interpolation NON-ADDITIVE of offsets
            for idx in range(len(anchors)):
                pos, ox, oy = anchors[idx]
                
                left_bound = pos - param.window
                if idx > 0:
                    prev_pos = anchors[idx-1][0]
                    if pos - prev_pos <= 2 * param.window:
                        left_bound = prev_pos
                
                right_bound = pos + param.window
                if idx < len(anchors) - 1:
                    next_pos = anchors[idx+1][0]
                    if next_pos - pos <= 2 * param.window:
                        right_bound = next_pos
                
                # Smooth left side (Fade in)
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
                        
                # Fix exact impact point
                if 0 <= pos < n_frames:
                    offsets_x[pos] = ox; offsets_y[pos] = oy
                    
                # Smooth right side (Fade Out)
                if right_bound == pos + param.window:
                    for i in range(pos + 1, right_bound + 1):
                        if i >= n_frames: continue
                        u = (i - pos) / param.window
                        w = 0.5 * (1 + math.cos(math.pi * u))
                        offsets_x[i] = ox * w; offsets_y[i] = oy * w

            # Apply force
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

        # Calcul absolute time
        abs_t = [0.0] * len(bd)
        acc = 0.0
        for i, obj in enumerate(bd):
            acc += obj.t
            abs_t[i] = acc

        # Generate click intervals
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

        # Clean intervals (Apply Offset/Jitter)
        presses = []
        for n, (t0, t1) in enumerate(raw_presses):
            start_t = t0 + param.offset_ms + rng.gauss(0, param.jitter_ms)
            end_t = t1 + param.offset_ms
            
            if end_t <= start_t:
                end_t = start_t + 1.0

            key = K1 if (not param.alternate or n % 2 == 0) else K2
            presses.append([start_t, end_t, key])


        # Release for next click
        for n in range(len(presses) - 1):
            current_start, current_end, current_key = presses[n]
            next_start, next_end, next_key = presses[n + 1]
            
            if current_end >= next_start - 2.0:
                presses[n][1] = next_start - 2.0

        def transform(frames):
            valid = []
            vtimes = []
            acc = 0.0
            
            # Extract times of replay
            for idx, fr in enumerate(frames):
                if fr[0] == -12345:
                    continue
                acc += fr[0]
                valid.append(idx)
                vtimes.append(acc)
            
            # Remove all clicks
            for idx in valid:
                frames[idx][3] &= ~CLICK_MASK

            # Inject new generate click
            for start_t, end_t, key in presses:

                v0 = bisect.bisect_left(vtimes, start_t)
                hi = bisect.bisect_right(vtimes, end_t)
                
                # Add a frame in replay for the click
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

        # Calcul absolute time
        abs_t = [0.0] * len(bd)
        acc = 0.0
        for i, obj in enumerate(bd):
            acc += obj.t
            abs_t[i] = acc

        # Waypoints generation
        wp_t, wp_x, wp_y = [], [], []
        prev_t = -1.0
        
        for idx, obj in enumerate(bd):
            if obj.object_type == 4:
                continue  # Don't process spinners here

            t_abs = abs_t[idx]
            
            # Offset 1.0 if two object are stack
            if t_abs <= prev_t:
                t_abs = prev_t + 1.0

            dt = t_abs - prev_t if prev_t >= 0 else 1000.0
            
            # Reduce spread during streaming part
            current_spread = param.spread
            if dt < 120:
                current_spread *= max(0.1, dt / 120.0)
                
            # Reduce spread at beggining and end of a slider
            if obj.object_type in (2, 3):
                current_spread *= 0.2

            ox = obj.x + rng.gauss(0, current_spread)
            oy = obj.y + rng.gauss(0, current_spread)
            
            wp_t.append(t_abs)
            wp_x.append(ox)
            wp_y.append(oy)
            
            prev_t = t_abs

        # Spinners
        spinners, k = [], 0
        while k < len(bd):
            if bd[k].object_type == 4:
                start = abs_t[k]
                end = abs_t[k + 1] if k + 1 < len(bd) and bd[k + 1].object_type == 4 else start
                spinners.append((start, end, bd[k].x, bd[k].y))
                k += 2
            else:
                k += 1
        
        # Inerpolation
        def hermite(p0, p1, p2, p3, u, cs):
            m1 = cs * (p2 - p0) * 0.5
            m2 = cs * (p3 - p1) * 0.5
            
            dist = abs(p2 - p1)
            max_tan = dist * 1.5 # Clamp frick
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

                # Spinner
                spin = next(((s0, cx, cy) for (s0, s1, cx, cy) in spinners if s0 <= T <= s1), None)
                if spin is not None:
                    s0, cx, cy = spin
                    ang = (T - s0) * param.spin_speed
                    fr[1] = round(cx + math.cos(ang) * param.spin_radius, 3)
                    fr[2] = round(cy + math.sin(ang) * param.spin_radius, 3)
                    continue
                
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
                
                u = (T - t0) / (t1 - t0) if t1 > t0 else 0.0

                i0, i3 = max(0, j - 1), min(n - 1, j + 2)
                fr[1] = round(hermite(wp_x[i0], wp_x[j], wp_x[j+1], wp_x[i3], u, param.curve_strength), 3)
                fr[2] = round(hermite(wp_y[i0], wp_y[j], wp_y[j+1], wp_y[i3], u, param.curve_strength), 3)

            return frames
    
        return patch_osr(raw, transform)

    def tap_assist(self, raw, bd):
        param = self.tap_assist_param

        def transform(frames):

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
        # elif self.tap_assist_param is not None:
        #     raw = self.tap_assist(raw, bd)

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


def process_cheat_file(osr_path: str, beatmap_path: str, output_folder: str, difficulty: CheatDifficulty = None):
    
    synth = CheatSynth()
    synth.init_replay(osr_path, beatmap_path, output_folder)
    
    synth.random_param(difficulty)
    synth.synth_cheat(skip_invalid=True)

def prepare_args(path: str, beatmap_folder: str):

    with open(path, 'rb') as f:
        replay_content = f.read()
    map_hash, _ = get_beatmap_replay_hash(replay_content)
    
    beatmap_path = os.path.join(beatmap_folder, f"{map_hash}.osu")
    if not os.path.exists(beatmap_path):
        return None
    
    return path, beatmap_path

def process_cheat_folder(
    replay_folder:  str, 
    beatmap_folder: str, 
    output_folder:  str, 
    max_workers:    int   = None,
    timeout:        float = 2.0,
    limit:          int   = 100,
    difficulty:     CheatDifficulty = None,
):
    print("Load dataset...")

    os.makedirs(output_folder, exist_ok=True)
    
    paths = [(os.path.join(replay_folder, filename), beatmap_folder)
             for filename in os.listdir(replay_folder)] 
    
    if limit is not None and limit < len(paths):
        paths = paths[:limit]
    
    args = []
    with ProcessPool(max_workers=max_workers or os.cpu_count()) as pool:
        futures = [pool.schedule(prepare_args, args=a, timeout=timeout) for a in paths]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = fut.result()
                if result is not None:
                    args.append((*result, output_folder, difficulty))
            except FutureTimeoutError:
                pass
            except Exception as e:
                pass
    
    print("Process dataset...")
    with ProcessPool(max_workers=max_workers or os.cpu_count()) as pool:
        futures = [pool.schedule(process_cheat_file, args=a, timeout=timeout) for a in args]
        for fut in tqdm(as_completed(futures), total=len(futures)):
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
    parser.add_argument("--difficulty", choices=["easy", "hard"], default=None, help="Cheat difficulty level (easy = blatant, hard = closet)")

    args = parser.parse_args()

    diff_val = None
    if args.difficulty == "easy":
        diff_val = CheatDifficulty.Easy
    elif args.difficulty == "hard":
        diff_val = CheatDifficulty.Hard

    process_cheat_folder(
        args.replays_folder, 
        args.beatmaps_folder, 
        args.output_folder,
        max_workers=args.max_workers,
        timeout=2,
        limit=args.limit,
        difficulty=diff_val
    )

    #process_cheat_file(
    #    osr_path,
    #    beatmap_path,
    #    "temp/cheat.osr"
    #)
