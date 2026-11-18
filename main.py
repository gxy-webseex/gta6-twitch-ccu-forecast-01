"""gta6-twitch-ccu-forecast-01 entry: runtime-comparable launch-day ratios.

Forecasts each hour h = 2..24 of release day as hour 1's print times a ratio
drawn from past big game launches. The training set is built at run time from
the case file's own `comparables` (the final case ships 42 launches: the 38
in the kit plus the 4 practice launches); the vendored NAMES/LOGP1/RATIOS/
JUMP7 below (37 kit comparables with a complete release day) are the
fallback if the case carries no usable comparables.

Model (round 3): for each hour, the five log-ratio quantiles are
  0.75 * (log-average of four members: Gaussian-kernel-weighted empirical
          quantiles of ln(viewers_h / viewers_h1) at bandwidths 1.0, 1.2
          and 1.5 on ln(p1), plus a local-linear median with kernel spread
          of residuals, bandwidth 1.2)
  + 0.25 * (2D-kernel quantiles on ln(p1) x jump7, bandwidths 1.2 / 1.2),
where jump7 = ln(hour-1 viewers / pre-release baseline), the baseline
being the median hourly viewers over the 7 days before release day. Then
the q05/q95 log-deviations from the median are narrowed by 0.8 (the
empirical tails are over-wide for the small effective samples). LOO
pinball: big 0.1977 / huge 0.2061 / all 0.4004 (v2: 0.2034 / 0.2126 /
0.4021); forward split (train p1<300k, test p1>=300k): 0.1452 (v2: 0.1595);
90% coverage on the big pool 0.874 (v2: 0.915). See MODEL_NOTES.md for the
full evaluation.

Run as the grader runs it: `python3 -P -s /bot/main.py`, standard library
only, no network, deterministic. Reads $MAKEFUN_INPUT_DIR (else /input) and
writes $MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
BWS = (1.0, 1.2, 1.5)
B1_WEIGHT = 0.25
B1_BW = 1.2
B1_B2 = 1.2
TAIL_MULT = 0.8
MIN_RUNTIME_LAUNCHES = 5

# Fallback training launches (37 kit comparables with a complete release day).
NAMES = ['ELDEN RING', 'Resident Evil Requiem', 'Hogwarts Legacy', 'Cyberpunk 2077', 'Ghost of Yōtei', 'Monster Hunter Wilds', 'Black Myth: Wukong', 'Hollow Knight: Silksong', 'Battlefield 6', 'Call of Duty: Black Ops 6', 'Call of Duty: Modern Warfare III', 'Call of Duty: Black Ops 7', 'Call of Duty: Modern Warfare 4', '007 First Light', 'Forza Horizon 6', 'Marathon', 'Saros', 'The Legend of Zelda: Tears of the Kingdom', 'Starfield', "Baldur's Gate 3", 'Diablo IV', 'Borderlands 4', 'Kingdom Come: Deliverance II', "Assassin's Creed Shadows", 'DOOM: The Dark Ages', 'Elden Ring Nightreign', 'Clair Obscur: Expedition 33', 'Death Stranding 2: On the Beach', 'FINAL FANTASY VII REBIRTH', 'The Last of Us Part II', 'Ghost of Tsushima', 'Halo Infinite', 'Forza Horizon 5', 'Stellar Blade', 'ARC Raiders', 'Marvel Rivals', 'HELLDIVERS 2']
# ln(hour-1 viewers) of each fallback training launch.
LOGP1 = [13.682156, 10.743869, 12.867105, 13.655341, 10.211193, 11.933699, 8.505323, 5.099866, 7.811973, 11.842927, 11.647605, 11.279820, 10.988035, 11.758222, 10.177780, 7.239215, 8.764053, 11.774890, 11.962905, 8.145260, 13.187130, 11.802366, 6.025866, 11.898140, 10.826774, 12.675792, 9.272282, 10.650673, 10.887923, 12.807705, 11.789875, 9.929448, 10.689556, 10.205701, 7.251345, 12.019376, 6.037871]
# RATIOS[i][j] = ln(viewers at hour j+2 / viewers at hour 1) of launch i.
RATIOS = [
    [-0.087983, -0.219442, -0.374617, -0.420986, -0.457004, -0.500824, -0.593712, -0.638563, -0.633950, -0.589605, -0.565143, -0.480838, -0.369715, -0.278450, -0.190638, -0.117068, -0.089611, -0.049289, -0.110815, -0.106559, -0.154390, -0.250298, -0.325551],
    [-0.142495, -0.336943, -0.105804, 0.197508, 1.250340, 1.519147, 1.527514, 1.448450, 1.439713, 1.488300, 1.571623, 1.710048, 1.840020, 1.954107, 2.147289, 2.212501, 2.305883, 2.448715, 2.496830, 2.440504, 2.307763, 2.213599, 2.107339],
    [-0.120046, -0.477328, -0.631352, -0.679547, -0.752249, -0.837369, -0.752090, -0.834404, -0.762140, -0.736623, -0.666421, -0.494188, -0.181657, -0.100628, -0.130624, 0.245330, 0.265199, 0.230474, 0.147192, 0.184169, 0.200710, 0.213165, 0.107580],
    [0.227154, 0.155737, -0.002313, -0.197141, -0.347290, -0.536919, -0.664654, -0.797190, -0.839967, -0.819135, -0.748500, -0.557666, -0.412049, -0.281022, -0.082069, -0.042670, -0.018965, -0.063978, -0.105593, -0.059739, -0.137474, -0.285211, -0.603971],
    [-0.366659, -1.193103, -0.872459, -0.219099, -0.089735, 0.039354, 0.145534, 0.281497, 0.282689, 0.289234, 0.383340, 0.676431, 0.844321, 0.978824, 0.999708, 1.017099, 1.280458, 1.261046, 1.238846, 1.285126, 1.253708, 1.169458, 1.237343],
    [-0.194579, -0.357461, -0.382785, -0.470626, 0.568477, 0.806911, 0.794308, 0.785574, 0.793851, 0.761156, 0.761806, 0.836337, 0.921759, 0.967256, 0.971875, 0.943505, 0.922670, 0.934272, 0.879873, 0.846036, 0.846951, 0.802510, 0.734786],
    [1.141012, 3.268920, 3.592886, 3.594266, 3.555441, 3.585597, 3.592011, 3.604503, 3.603787, 3.643687, 3.797550, 4.005731, 4.114540, 4.232627, 4.209075, 4.336286, 4.396107, 3.843663, 3.657795, 4.396067, 4.522305, 4.364427, 4.182824],
    [-0.600057, -0.589007, -0.484746, -0.304076, -0.179886, 0.098631, 0.223144, 1.086342, 2.861852, 3.137877, 3.493732, 4.527670, 6.240906, 7.612717, 7.659439, 7.626832, 7.623329, 7.535915, 7.457426, 7.383099, 7.300218, 7.189761, 6.991990],
    [-0.022519, -0.069137, -0.112584, -0.329855, -0.487483, -0.220112, 0.433936, 0.823714, 1.110418, 1.415224, 1.683847, 2.317015, 3.086967, 4.033258, 5.270324, 5.396417, 5.418554, 5.430157, 5.326865, 5.191262, 5.052259, 4.882093, 4.736529],
    [-0.094115, -0.169627, -0.168801, 0.136684, 0.253961, 0.236261, 0.228442, 0.177942, 0.123002, 0.113177, 0.094364, 0.182565, 0.272570, 0.417591, 0.598898, 0.717254, 0.826433, 0.955964, 0.983980, 0.926717, 1.007749, 0.968126, 0.944294],
    [-0.107590, -0.174055, -0.287598, -0.291649, -0.010694, 0.111172, 0.146288, 0.083275, 0.076529, 0.081794, 0.074852, 0.045699, 0.106013, 0.230422, 0.201207, 0.241988, 0.300312, 0.355611, 0.380256, 0.476058, 0.539611, 0.521705, 0.461582],
    [-0.142141, -0.233911, -0.257787, -0.272699, -0.178946, -0.110921, -0.074575, -0.088630, -0.063187, -0.038572, -0.032841, 0.006330, 0.037480, 0.152275, 0.305035, 0.465795, 0.447822, 0.578792, 0.578736, 0.531275, 0.627017, 0.656384, 0.471224],
    [-0.030967, -0.091647, -0.160725, -0.141089, -0.047154, -0.021286, -0.018992, -0.001658, -0.048342, -0.074529, -0.022270, 0.209289, 0.398193, 0.508670, 0.622671, 0.668506, 0.725216, 0.876139, 0.863546, 0.872571, 0.882180, 0.864017, 0.816164],
    [-0.212607, -0.554965, -0.582463, -0.653580, -0.769477, -1.042161, -0.974361, -0.949322, -0.991781, -0.927662, -0.886629, -0.618958, -0.427418, -0.262483, -0.184248, -0.083356, 0.058741, 0.271442, 0.428413, 0.376409, 0.237093, 0.049051, 0.155103],
    [-0.285809, -0.276946, -0.304443, -0.123161, -0.021047, -0.042308, -0.049990, -0.001598, 0.102155, 0.220922, 0.374066, 0.526767, 0.718681, 0.796006, 0.947451, 0.883406, 0.932426, 1.197938, 1.565201, 1.666799, 1.501998, 1.450757, 0.893115],
    [-0.257280, -0.958819, -0.820850, -1.232862, -1.220622, -1.232862, -1.166170, -1.065429, -0.983465, -0.936596, -0.820850, -0.475330, -0.105121, 0.533117, 1.957838, 2.565777, 3.427855, 4.713081, 4.798877, 4.739085, 4.697074, 4.522927, 4.442136],
    [-0.137647, -0.282902, -0.310013, -0.486387, -0.275671, 0.137586, 0.249177, 0.269192, 0.136087, 0.493266, 0.526853, 0.952984, 1.152499, 1.477833, 1.619914, 1.569494, 1.690874, 1.728220, 1.645016, 1.612901, 1.600776, 1.282538, 1.043034],
    [-0.182894, -0.405592, -0.669376, 0.074129, 0.152011, 0.141906, 0.058357, 0.025565, 0.145625, 0.248713, 0.395722, 0.493665, 0.648956, 0.806869, 0.816401, 0.929622, 0.956243, 0.980122, 0.972041, 0.909756, 0.832225, 0.780328, 0.643067],
    [0.058974, 0.048751, -0.085059, -0.482978, -0.560933, -0.670826, -0.583008, -0.700790, -0.855590, -1.028406, -1.003991, -0.525371, -0.547912, -0.542084, -0.593123, -0.551282, -0.472082, -0.370420, -0.349971, -0.432916, -0.421635, -0.323796, -0.348922],
    [-0.271281, -0.237976, -0.535893, -1.019976, -1.373324, -0.900318, -0.505617, -0.061314, 0.118846, 0.254725, 0.647442, 2.379933, 2.985175, 3.681608, 4.462307, 4.800407, 4.900067, 4.894285, 4.881513, 4.810865, 4.637904, 4.530835, 4.357835],
    [-0.004711, -0.057045, -0.120554, -0.186351, -0.262630, -0.272081, -0.203167, -0.150368, -0.092485, -0.059697, -0.012732, -0.006970, 0.042497, 0.059209, 0.107079, 0.184709, 0.233570, 0.223768, 0.212171, 0.183789, 0.136689, 0.052965, -0.041079],
    [0.517834, 0.072505, -0.354011, -0.391728, -0.521726, -0.541254, -0.594418, -0.730149, -0.871136, -0.699019, -0.651631, -0.678491, -0.569419, -0.437975, -0.371269, -0.280996, -0.207732, -0.013705, 0.032389, 0.048438, -0.072098, -0.002301, 0.152099],
    [0.294902, 0.560306, 0.765355, 0.935430, 1.043157, 1.124051, 1.364315, 2.090850, 2.687058, 2.749065, 3.309343, 3.621761, 3.802413, 4.102683, 5.393441, 6.531392, 6.802550, 6.850226, 6.895406, 7.007358, 6.880650, 6.646591, 6.467057],
    [0.020424, -0.382518, -0.919326, -1.000235, -1.039026, -1.182701, -1.073753, -0.963214, -0.756626, -0.664638, -0.558937, -0.420583, -0.299624, -0.302170, -0.302189, 0.022967, -0.063023, -0.029809, -0.012011, -0.149747, -0.157601, 0.045184, -0.123582],
    [-0.022576, -0.354739, -0.704593, -0.762486, -0.742257, -0.697107, -0.513232, -0.522030, -0.677051, -0.703067, -0.545261, -0.360248, -0.090986, 0.152620, 0.243483, 0.322884, 0.328834, 0.493295, 0.304862, 0.163304, 0.051424, -0.040883, 0.040689],
    [-0.127293, -0.367253, -0.272287, -0.440437, -0.676215, -0.728438, -0.707418, -0.731980, -0.737664, -0.692850, -0.718559, -0.631380, -0.523231, -0.407919, -0.241278, -0.114429, -0.021214, -0.003306, -0.054030, -0.078275, -0.097032, -0.132879, -0.199085],
    [-0.274763, -0.754889, -0.641581, -0.655424, -0.934694, -0.412492, 0.675031, 0.939940, 1.177041, 1.191022, 1.355924, 1.673018, 1.936981, 2.099483, 2.073538, 2.133537, 2.162423, 2.319538, 2.410446, 2.341376, 2.293046, 2.116089, 1.945239],
    [-0.237991, -0.356191, -0.291788, -0.242599, -0.237691, -0.231581, -0.215117, -0.313424, -0.361651, -0.429678, -0.265191, -0.070981, 0.127429, 0.332720, 0.462879, 0.435553, 0.536785, 0.488620, 0.434557, 0.401549, 0.334382, 0.039247, -0.007775],
    [0.060460, -0.098130, -0.193164, -0.178942, 0.108494, 0.044845, -0.059561, -0.186253, -0.242593, -0.164655, -0.081554, 0.057448, 0.166073, 0.282132, 0.333794, 0.290159, 0.321123, 0.404368, 0.705411, 0.806007, 0.767552, 0.596799, 0.270098],
    [-0.303600, -0.623643, -0.786842, -0.343006, -0.404993, -0.511275, -0.599569, -0.682598, -0.720494, -0.684356, -0.533114, -0.335368, -0.182432, -0.021340, 0.069816, 0.180309, 0.215684, 0.269758, 0.291040, 0.244740, 0.138415, 0.154319, 0.155780],
    [-0.193041, -0.324849, -0.433767, -0.059213, 0.066909, -0.056450, -0.106642, -0.125887, -0.164175, -0.122959, -0.089044, 0.127301, 0.325353, 0.440339, 0.502242, 0.571349, 0.668115, 0.717294, 0.808736, 0.812420, 0.777188, 0.740977, 0.712249],
    [0.205509, 0.206658, 0.247458, 0.220783, 0.177919, 0.133732, -0.281175, -0.532877, -0.515187, -0.745246, -0.945255, -1.243187, -1.202642, -1.251327, -1.046640, -0.664524, 0.444981, 1.734051, 2.159591, 2.164933, 1.944350, 1.812643, 1.565517],
    [-0.362436, -0.718316, -0.821229, -1.209265, -1.156335, -0.806067, -0.709570, -0.740999, -0.455398, -0.244180, -0.075769, -0.120035, -0.307043, -0.252298, 0.097989, 0.205683, 0.310976, 0.096976, 0.245923, 0.278004, 0.468579, 0.662532, 0.172650],
    [-0.245455, -0.741029, -0.655750, -0.074321, -0.004891, -0.460038, -0.541423, -0.297524, -0.197493, -0.146535, 0.027809, 0.109333, 0.263299, 0.754204, 1.001555, 0.952875, 0.969946, 1.223680, 1.177526, 1.158504, 1.128708, 1.029928, 0.903473],
    [-0.267555, 0.076436, 0.243641, 0.268890, 0.206264, 0.787812, 1.590248, 2.494377, 4.203358, 4.988230, 5.156599, 5.260244, 5.339071, 5.356128, 5.413607, 5.505257, 5.519021, 5.559787, 5.540231, 5.451848, 5.457761, 5.430269, 5.358521],
    [0.271939, 0.336491, 0.235797, 0.076090, -0.018772, -0.187531, -0.315640, -0.427768, -0.581346, -0.746320, -0.733551, -0.692069, -0.555998, -0.461783, -0.369492, -0.279641, -0.195845, -0.168194, -0.182058, 0.374417, 0.475808, 0.119659, -0.171779],
    [-0.492693, -1.707138, -1.560534, -1.463160, -0.956467, -0.288478, 0.168705, 2.083609, 4.092912, 4.496942, 4.570000, 4.732696, 4.961408, 5.147592, 5.199670, 5.111887, 5.416758, 5.489460, 5.775826, 5.848843, 5.823547, 5.714127, 5.658301],
]
# JUMP7[i] = ln(hour-1 viewers / pre-release baseline) of launch i.
JUMP7 = [6.863232, 5.990278, 1.526279, 7.776205, 7.266754, 7.180109, 5.286447, 3.713572, 3.961826, 5.467051, 3.212924, 6.017130, 1.017379, 8.925008, 0.455036, -2.797535, -0.459697, 6.747725, 0.229403, 1.020781, 0.579706, 7.471633, 5.332719, 9.595555, 6.976626, 8.783972, 5.488092, 7.095325, 5.111820, 9.144143, 7.194755, 0.121645, 1.029670, 6.679341, 1.311174, 7.211264, 3.512142]


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript Math.round(x*100)/100)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def baseline_from_points(points, day_start, days=7):
    """Median hourly viewers over the `days` days before day_start.

    Points carry 'from' (comparable hourly / history_hourly) or 'at'
    (history) timestamps; they are bucketed by hour (a mean per hour, so
    minute-level histories count once per hour), zero hours are dropped,
    and the median over the hours in the window is returned, or None when
    fewer than 5 usable hours exist.
    """
    by_hour = {}
    for r in points or []:
        try:
            ts = parse_ts(r.get("from") or r.get("at"))
            v = float(r.get("viewers") or 0)
        except (TypeError, ValueError):
            continue
        if not (day_start - timedelta(days=days) <= ts < day_start):
            continue
        h = ts.replace(minute=0, second=0, microsecond=0)
        by_hour.setdefault(h, []).append(v)
    vals = []
    for h in sorted(by_hour):
        m = sum(by_hour[h]) / len(by_hour[h])
        if m > 0:
            vals.append(m)
    if len(vals) < 5:
        return None
    return statistics.median(vals)


def parse_comparables(case):
    """Build the training set from the case's own comparables.

    Returns a list of (logp1, ratios[23], jump7) sorted by launch name,
    keeping only launches with a complete release day (all 24 UTC hours
    present, viewers > 0). jump7 is None for a launch whose pre-release
    baseline cannot be computed; such launches still train the other
    members. Returns [] when the case carries no usable comparables.
    """
    out = []
    for x in case.get("comparables") or []:
        try:
            ds = parse_ts(x["day_start"])
            by = {parse_ts(r["from"]): r for r in x["hourly"]}
        except (KeyError, TypeError, ValueError):
            continue
        hrs = []
        ok = True
        for h in range(1, 25):
            r = by.get(ds + timedelta(hours=h - 1))
            if r is None or not r.get("viewers"):
                ok = False
                break
            hrs.append(r["viewers"])
        if not ok:
            continue
        p1 = hrs[0]
        base = baseline_from_points(x.get("hourly"), ds)
        jump = math.log(p1 / base) if base else None
        out.append((x.get("name") or "", math.log(p1),
                    [math.log(v / p1) for v in hrs[1:]], jump))
    out.sort(key=lambda t: t[0])
    return [(lp, ratios, jump) for _, lp, ratios, jump in out]


def fallback_train():
    return list(zip(LOGP1, RATIOS, JUMP7))


def test_jump(case, p1):
    """The test launch's jump7 from the case's own pre-release history.

    Reads history_hourly (preferred) or history over the 7 days before the
    case's day_start (checkpoint minus one hour when day_start is absent).
    Returns None when no usable baseline exists.
    """
    try:
        ds = parse_ts(case["day_start"])
    except (KeyError, TypeError, ValueError):
        try:
            ds = parse_ts(case["checkpoint"]) - timedelta(hours=1)
        except (KeyError, TypeError, ValueError):
            return None
    pts = case.get("history_hourly") or case.get("history") or []
    base = baseline_from_points(pts, ds)
    return math.log(p1 / base) if base else None


def wquantile(pairs, tau):
    """Weighted empirical quantile (linear interpolation on cumulative weight).

    pairs is a list of (value, weight) sorted by value, then weight.
    """
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc >= target:
            if i == 0 or w <= 0:
                return v
            prev, _ = pairs[i - 1]
            return prev + (v - prev) * max(0.0, (target - (acc - w)) / w)
    return pairs[-1][0]


def kernel_log_quantiles(train, lp1, j, bw):
    """Kernel-weighted empirical quantiles of log-ratios at hour index j."""
    pairs = sorted(
        (ratios[j], math.exp(-((lp - lp1) / bw) ** 2))
        for lp, ratios, _ in train
    )
    return [wquantile(pairs, tau) for tau in LEVELS]


def kernel_2d_log_quantiles(train, lp1, jump, j, bw=B1_BW, b2=B1_B2):
    """2D-kernel quantiles on (ln(p1), jump7). Returns None when fewer than
    5 training launches carry a jump."""
    pairs = []
    for lp, ratios, jp in train:
        if jp is None:
            continue
        w = (math.exp(-((lp - lp1) / bw) ** 2)
             * math.exp(-((jp - jump) / b2) ** 2))
        pairs.append((ratios[j], w))
    if len(pairs) < 5:
        return None
    pairs.sort()
    return [wquantile(pairs, tau) for tau in LEVELS]


def lin_slope(train, lp1, j, bw=1.2):
    """Kernel-weighted linear regression of log-ratio on ln(p1), centred at
    lp1. Returns (slope, intercept_at_lp1)."""
    sw = sx = sy = sxx = sxy = 0.0
    for lp, ratios, _ in train:
        w = math.exp(-((lp - lp1) / bw) ** 2)
        dx = lp - lp1
        y = ratios[j]
        sw += w
        sx += w * dx
        sy += w * y
        sxx += w * dx * dx
        sxy += w * dx * y
    den = sw * sxx - sx * sx
    if den <= 1e-12 or sw <= 0:
        return 0.0, (sy / sw if sw > 0 else 0.0)
    slope = (sw * sxy - sx * sy) / den
    return slope, (sy - slope * sx) / sw


def loclin_log_quantiles(train, lp1, j, bw=1.2):
    """Local-linear median + kernel spread of residuals around the fit."""
    slope, med = lin_slope(train, lp1, j, bw)
    resid = sorted(
        (ratios[j] - (med + slope * (lp - lp1)),
         math.exp(-((lp - lp1) / bw) ** 2))
        for lp, ratios, _ in train
    )
    return [med + wquantile(resid, tau) for tau in LEVELS]


def hour_quantiles(train, p1, h, jump):
    """The five quantiles of hour h's viewers, h = 2..24.

    0.75 * log-average of the four round-2 members + 0.25 * the 2D-kernel
    member on (ln(p1), jump7), then the q05/q95 log-deviations from the
    median are narrowed by TAIL_MULT. When jump is None (or too few
    training launches carry one) the 2D member is skipped.
    """
    j = h - 2
    lp1 = math.log(p1)
    members = [kernel_log_quantiles(train, lp1, j, bw) for bw in BWS]
    members.append(loclin_log_quantiles(train, lp1, j))
    ship = [sum(m[k] for m in members) / len(members) for k in range(5)]
    if jump is not None:
        b1 = kernel_2d_log_quantiles(train, lp1, jump, j)
        if b1 is not None:
            ship = [(1.0 - B1_WEIGHT) * ship[k] + B1_WEIGHT * b1[k]
                    for k in range(5)]
    med = ship[2]
    logq = [med + TAIL_MULT * (ship[0] - med), ship[1], med, ship[3],
            med + TAIL_MULT * (ship[4] - med)]
    out = []
    prev = 0.01
    for lq in logq:
        q = round2(p1 * math.exp(lq))
        q = min(max(q, 0.01), 1e9)
        q = max(q, prev)  # keep the row non-decreasing after rounding/clamping
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None:
        try:
            p1 = max(1.0, float(first.get("viewers") or 1))
        except (TypeError, ValueError):
            p1 = 1.0
    train = parse_comparables(case)
    if len(train) < MIN_RUNTIME_LAUNCHES:
        train = fallback_train()
    jump = test_jump(case, p1)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = []
    for row in rows:
        h = int(row["hour"])
        hours.append({"hour": h, "q": hour_quantiles(train, p1, h, jump)})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
