from osrparse import Mod

SUPPORTED_MODS = Mod.NoFail      | Mod.Easy    | Mod.HalfTime   | Mod.HardRock   | \
                 Mod.SuddenDeath | Mod.Perfect | Mod.DoubleTime | Mod.Nightcore  | \
                 Mod.Flashlight  | Mod.Hidden

def is_supported(mods) -> bool:
    return not (mods & ~SUPPORTED_MODS)

def get_rate(mods) -> float:
    if mods & (Mod.DoubleTime | Mod.Nightcore):
        return 1.5
    if mods & Mod.HalfTime:
        return 0.75
    return 1.0

def apply_mods_to_difficulty(cs: float, od: float, mods) -> tuple[float, float]:
    if mods & Mod.HardRock:
        return min(cs * 1.3, 10), min(od * 1.4, 10) # refer to: https://osu.ppy.sh/wiki/en/Gameplay/Game_modifier/Hard_Rock 
    if mods & Mod.Easy:                             
        return cs * 0.5, od * 0.5                   # refer to: https://osu.ppy.sh/wiki/en/Gameplay/Game_modifier/Easy
    return cs, od 