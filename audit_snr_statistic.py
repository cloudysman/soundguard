"""What the frame-percentile statistic of the rejection policy actually measures.

The recording axis of the shipped rejection policy divides the ninetieth percentile of frame energy
by the tenth and calls the result an estimate of the signal-to-noise ratio. This script establishes
what that quantity is, and what follows from it.

The argument runs from an algebraic identity to its consequences, so the central claim needs no
dataset and cannot be answered by doubting one.

    part 1   the identity      the statistic equals the interquantile range of frame energy in
                               decibels, so it is a measure of dispersion and not of noise
    part 2   gain              a dispersion measure is invariant to gain, so the statistic cannot
                               see a recording made quieter
    part 3   silence           removing signal raises the statistic, so a damaged recording is
                               scored as a cleaner one
    part 4   the floor         once part of a recording is silent the statistic reports the
                               regularisation constant, so its magnitude is not reportable
    part 5   imposed noise     on real recordings at a known signal-to-noise ratio, the statistic
                               follows that ratio only weakly, and more weakly for continuous sounds
    part 6   consequence       the rejection rate differs across categories by an order of magnitude
    part 7   limiting case     a synthetic sweep, included because it isolates the mechanism, not
                               because it models real audio

Two terms are used throughout, each in one sense only. The frame-percentile statistic is the
quantity under audit. The frame-energy dispersion is what it measures.

Every conclusion printed below is derived from the numbers computed immediately above it. None is
written into the source. If the data change, the verdicts change with them.

    python audit_snr_statistic.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
DURATION_S = 4
FRAME_MS = 25
FLOOR = 1e-12                  # matches the constant in corruption.py
REJECT_THRESHOLD_DB = 3.0      # the severe threshold of the audited policy
PERCENTILES = (90, 10)

QUALITY_TABLE = "rerun_results/quality.csv"
DEMO_DIR = "samples"

CATEGORIES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]
# Categories whose sound is present throughout a recording rather than in bursts.
CONTINUOUS = {"air_conditioner", "drilling", "engine_idling", "jackhammer", "siren", "street_music"}
# Clean accuracy per category, from Table 6.1 of the report.
CLEAN_ACCURACY = {
    "gun_shot": 91.7, "street_music": 78.9, "dog_bark": 76.7, "children_playing": 74.2,
    "siren": 73.2, "drilling": 72.9, "engine_idling": 68.1, "jackhammer": 58.1,
    "car_horn": 57.8, "air_conditioner": 55.6,
}
# gaussian_noise severity to the signal-to-noise ratio it imposes, from corruption.py.
IMPOSED_SNR_DB = {1: 20, 2: 15, 3: 10, 4: 5, 5: 0}


# --------------------------------------------------------------------------- the statistic

def frame_energy(wav: np.ndarray, sr: int, floor: float = FLOOR) -> np.ndarray:
    """Mean energy of each non-overlapping frame, with the regularisation constant added."""
    n = max(1, int(sr * FRAME_MS / 1000))
    k = len(wav) // n
    if k < 2:
        return np.array([float(np.mean(wav ** 2)) + floor])
    return np.mean(wav[: k * n].reshape(k, n) ** 2, axis=1) + floor


def statistic(wav: np.ndarray, sr: int, percentiles=PERCENTILES, floor: float = FLOOR) -> float:
    """The frame-percentile statistic, in decibels."""
    hi, lo = percentiles
    e = frame_energy(wav, sr, floor)
    return float(10 * np.log10(np.percentile(e, hi) / np.percentile(e, lo)))


def dispersion(wav: np.ndarray, sr: int, percentiles=PERCENTILES, floor: float = FLOOR) -> float:
    """The interquantile range of frame energy expressed in decibels."""
    hi, lo = percentiles
    e_db = 10 * np.log10(frame_energy(wav, sr, floor))
    return float(np.percentile(e_db, hi) - np.percentile(e_db, lo))


def quiet_fraction(wav: np.ndarray, sr: int, below_peak_db: float = 20.0) -> float:
    """Fraction of frames lying more than ``below_peak_db`` under the loudest frame."""
    e = frame_energy(wav, sr)
    return float(np.mean(10 * np.log10(e / e.max()) < -below_peak_db))


# --------------------------------------------------------------------------- helpers

def load_demo_recordings(directory: str = DEMO_DIR) -> list[tuple[str, np.ndarray, int]]:
    try:
        import soundfile as sf
    except ImportError:
        return []
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".wav"):
            wav, sr = sf.read(os.path.join(directory, name), dtype="float64", always_2d=True)
            out.append((name[:-4], wav.mean(axis=1), sr))
    return out


def load_quality_table(path: str = QUALITY_TABLE):
    try:
        import pandas as pd
    except ImportError:
        return None
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path)
    table["category"] = table.classID.map(lambda i: CATEGORIES[i])
    table["continuous"] = table.category.isin(CONTINUOUS)
    return table


def heading(number: int, title: str) -> None:
    print(f"\n\nPart {number}. {title}\n" + "-" * 74)


def skipped(number: int, title: str, why: str) -> None:
    print(f"\n\nPart {number}. {title}\n  skipped: {why}")


# --------------------------------------------------------------------------- part 1

def part_identity() -> None:
    heading(1, "The statistic is the frame-energy dispersion")
    recordings = load_demo_recordings()
    if not recordings:
        print("  skipped: no demonstration recordings available.")
        return

    print(f"  {'recording':12s} {'statistic':>12s} {'dispersion':>13s} {'difference':>13s}")
    gaps = []
    for name, wav, sr in recordings:
        a, b = statistic(wav, sr), dispersion(wav, sr)
        gaps.append(abs(a - b))
        print(f"  {name:12s} {a:9.3f} dB {b:10.3f} dB {abs(a - b):13.2e}")

    print(f"\n  largest difference: {max(gaps):.2e} dB")
    if max(gaps) < 1e-2:
        print("  The two quantities agree to within arithmetic error, as the algebra requires: the")
        print("  difference of two logarithms is the logarithm of their quotient. The statistic is")
        print("  therefore a measure of dispersion, and no measurement can make it a measure of noise.")
    else:
        print("  The two quantities do not agree, so the algebraic identity does not hold here.")


# --------------------------------------------------------------------------- part 2

def part_gain() -> None:
    heading(2, "A dispersion measure cannot see gain")
    recordings = load_demo_recordings()
    if recordings:
        gains = (1.0, 0.5, 0.1, 0.06, 0.01)
        name, wav, sr = recordings[0]
        values = [statistic(wav * g, sr) for g in gains]
        print(f"  recording '{name}' scaled by {', '.join(str(g) for g in gains)}:")
        print("    " + "  ".join(f"{v:.3f} dB" for v in values))
        if max(values) - min(values) < 1e-2:
            print(f"\n  The statistic does not move across a {max(gains) / min(gains):.0f}-fold "
                  "range of gain.")
            print("  A recording-quality gate exists in part to catch a recording made too quiet.")
            print("  This statistic cannot do so, which is why the policy carries a separate level")
            print("  measurement beside it.")
        else:
            print("\n  The statistic does move with gain.")

    table = load_quality_table()
    if table is None:
        return
    quiet = table[table.corruption == "quiet"]
    if quiet.empty:
        return
    delta = (quiet.snr_corr - quiet.snr_clean).abs()
    print(f"\n  On real recordings, the corruption named quiet multiplies the waveform by a "
          f"constant.")
    print(f"  Across {len(quiet)} recordings the statistic changes by a median of "
          f"{delta.median():.3f} dB,")
    print(f"  and {100 * (delta < 0.05).mean():.1f} percent of them change by less than 0.05 dB.")


# --------------------------------------------------------------------------- part 3

def part_silence() -> None:
    heading(3, "Removing signal raises the statistic")
    table = load_quality_table()
    if table is None:
        print(f"  skipped: {QUALITY_TABLE} not available.")
        return

    print(f"  {'corruption':18s} {'n':>5s} {'median change':>15s} {'rejection rate':>22s}")
    raised = []
    for name in sorted(table.corruption.dropna().unique()):
        s = table[table.corruption == name]
        change = float((s.snr_corr - s.snr_clean).median())
        before = float((s.snr_clean < REJECT_THRESHOLD_DB).mean() * 100)
        after = float((s.snr_corr < REJECT_THRESHOLD_DB).mean() * 100)
        print(f"  {name:18s} {len(s):5d} {change:+12.1f} dB {before:12.1f}% -> {after:5.1f}%")
        if change > 1.0 and after < before:
            raised.append((name, change, before, after))

    if raised:
        print("\n  These corruptions both raise the statistic and lower the rejection rate:")
        for name, change, before, after in raised:
            print(f"    {name}: rejection rate {before:.1f} percent to {after:.1f} percent")
        print("  Removing part of a recording therefore makes the policy treat it as cleaner.")
        print("  Report the rejection rate rather than the change in decibels; part 4 shows why.")
    else:
        print("\n  No corruption both raises the statistic and lowers the rejection rate.")


# --------------------------------------------------------------------------- part 4

def part_floor(severity: int = 5) -> None:
    heading(4, "The magnitude reports the regularisation constant")
    recordings = load_demo_recordings()
    try:
        import corruption as C
    except ImportError:
        print("  skipped: corruption.py not importable.")
        return
    if not recordings:
        print("  skipped: no demonstration recordings available.")
        return

    floors = (1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)
    print("  a recording with part of its length silenced, scored at several floor constants\n")
    print("  " + " " * 12 + "".join(f"{f:>10.0e}" for f in floors))
    values = []
    for name, wav, sr in recordings:
        masked = C.corrupt(wav, sr, "time_mask", severity, seed=0)
        row = [statistic(masked, sr, floor=f) for f in floors]
        values.append(row)
        print(f"  {name:12s}" + "".join(f"{v:10.1f}" for v in row))

    arr = np.array(values)
    span = float(np.mean(arr[:, 0] - arr[:, -1]))
    decades = abs(np.log10(floors[-1] / floors[0]))
    rejected = [int((arr[:, j] < REJECT_THRESHOLD_DB).sum()) for j in range(len(floors))]

    print(f"\n  across {decades:.0f} orders of magnitude the statistic moves {span:.1f} dB, "
          f"or {span / decades:.1f} dB per order")
    print(f"  recordings rejected at each floor: {rejected}")
    if abs(span) > 5.0 and len(set(rejected)) == 1:
        print("\n  Once part of a recording is silent the low percentile is the floor constant, so")
        print("  the statistic reports that constant and not the recording. Its magnitude is")
        print("  therefore not reportable. The decision is: it does not change across the range.")
    elif abs(span) > 5.0:
        print("\n  Both the magnitude and the decision move with the floor constant.")
    else:
        print("\n  The statistic is insensitive to the floor constant.")


# --------------------------------------------------------------------------- part 5

def part_imposed_noise() -> None:
    heading(5, "Response to noise imposed at a known ratio")
    table = load_quality_table()
    if table is None:
        print(f"  skipped: {QUALITY_TABLE} not available.")
        return
    subset = table[table.corruption == "gaussian_noise"]
    if subset.empty:
        print("  skipped: no recordings carry the corruption named gaussian_noise.")
        return

    print(f"  {'imposed ratio':>14s} {'continuous':>13s} {'intermittent':>15s}")
    for severity in sorted(IMPOSED_SNR_DB, reverse=True):
        s = subset[subset.severity == severity]
        print(f"  {IMPOSED_SNR_DB[severity]:11d} dB {s[s.continuous].snr_corr.median():10.1f} dB "
              f"{s[~s.continuous].snr_corr.median():12.1f} dB")

    imposed_span = IMPOSED_SNR_DB[1] - IMPOSED_SNR_DB[5]
    moves = {}
    for label, group in (("continuous", subset[subset.continuous]),
                         ("intermittent", subset[~subset.continuous])):
        high = group[group.severity == 1].snr_corr.median()
        low = group[group.severity == 5].snr_corr.median()
        moves[label] = float(high - low)

    print(f"\n  when the imposed ratio falls by {imposed_span} dB the statistic falls by")
    for label, move in moves.items():
        print(f"    {label:13s} {move:5.1f} dB, which is {move / imposed_span * 100:3.0f} percent "
              "of the fall a true estimate would show")
    weaker = min(moves, key=moves.get)
    ratio = max(moves.values()) / max(min(moves.values()), 1e-9)
    print(f"\n  Both groups follow the imposed ratio weakly. The {weaker} group follows it "
          f"{ratio:.1f} times")
    print("  less closely than the other, which is the difference the rejection rate inherits.")


# --------------------------------------------------------------------------- part 6

def part_categories() -> None:
    heading(6, "The consequence: rejection falls unevenly across categories")
    table = load_quality_table()
    if table is None:
        print(f"  skipped: {QUALITY_TABLE} not available.")
        return
    try:
        from scipy import stats as sps
    except ImportError:
        print("  skipped: scipy not available.")
        return

    unique = table.drop_duplicates(subset="file")
    # Two rejection rules, reported side by side because they answer different questions. The
    # statistic alone isolates the quantity this script audits, and it is what Table 7.3 and
    # Figure 7.3 of the report state, so that the table and the figure share one source. The whole
    # recording axis is what the deployed policy applies, since the axis also carries a level test
    # and a clipping test joined by or; it is reported here for comparison, not used in the report.
    rules = {
        "statistic alone": unique.snr_clean < REJECT_THRESHOLD_DB,
        "whole recording axis": (unique.snr_clean < REJECT_THRESHOLD_DB)
                                | (unique.level_clean < -42.0)
                                | (unique.clip_clean > 30.0),
    }

    for rule_name, rejected in rules.items():
        frame = unique.assign(rejected=rejected)
        grouped = frame.groupby("category").agg(
            n=("file", "size"), median_statistic=("snr_clean", "median"), rate=("rejected", "mean")
        ).reset_index()
        grouped["rate"] *= 100
        grouped["accuracy"] = grouped.category.map(CLEAN_ACCURACY)
        grouped = grouped.sort_values("rate", ascending=False)

        print(f"\n  rejecting on the {rule_name}\n")
        print(f"  {'category':18s} {'n':>5s} {'median statistic':>18s} {'rejected':>10s} "
              f"{'accuracy':>10s}")
        for _, r in grouped.iterrows():
            print(f"  {r.category:18s} {r.n:5d} {r.median_statistic:15.1f} dB {r.rate:9.1f}% "
                  f"{r.accuracy:9.1f}%")

        spread = grouped.rate.max() / max(grouped.rate.min(), 1e-9)
        rho = sps.spearmanr(grouped.accuracy, grouped.rate)
        cont = grouped[grouped.category.isin(CONTINUOUS)].rate
        inter = grouped[~grouped.category.isin(CONTINUOUS)].rate
        mw = sps.mannwhitneyu(cont, inter, alternative="greater")

        print(f"\n    overall rejection rate: {frame.rejected.mean() * 100:.1f} percent")
        print(f"    spread across categories: {spread:.0f}-fold")
        print(f"    rank correlation with category accuracy: {rho.statistic:+.3f} "
              f"(p = {rho.pvalue:.4f})")
        print(f"    continuous {cont.mean():.1f} percent against intermittent "
              f"{inter.mean():.1f} percent (p = {mw.pvalue:.4f})")

        if rho.pvalue < 0.05 and rho.statistic < 0:
            print("    → rejection falls most on the categories the classifier reads least well.")
        elif rho.pvalue < 0.05:
            print("    → rejection falls most on the categories the classifier reads best.")
        else:
            print("    → no significant relation with category accuracy under this rule.")

    print("\n  Quote whichever rule the surrounding sentence is about, and say which it is. The")
    print("  two differ because the recording axis rejects on three tests joined by or, so it")
    print("  cannot be used to describe the behaviour of one of them.")


# --------------------------------------------------------------------------- part 7

DUTIES = (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00)
TRUE_SNRS_DB = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
SEEDS = range(20)


@dataclass(frozen=True)
class Sweep:
    duty: np.ndarray
    snr: np.ndarray
    value: np.ndarray

    def span_at(self, duty: float) -> float:
        """How far the statistic moves across the whole range of imposed ratios."""
        means = [self.value[(self.duty == duty) & (self.snr == s)].mean() for s in TRUE_SNRS_DB]
        return float(max(means) - min(means))


def synthesise(duty: float, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """A recording active for ``duty`` of its length at a known ratio.

    The active part is stationary, so this is the limiting case of a source with no intrinsic
    modulation. Real continuous sources do modulate, which is why their dispersion is not zero.
    """
    n = SAMPLE_RATE * DURATION_S
    active = np.zeros(n)
    span = int(round(duty * n))
    if span > 0:
        start = 0 if span == n else int(rng.integers(0, n - span + 1))
        carrier = rng.normal(0.0, 1.0, span)
        active[start:start + span] = carrier / (np.sqrt(np.mean(carrier ** 2)) + FLOOR)
    noise = rng.normal(0.0, 1.0, n)
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + FLOOR) * 10 ** (-snr_db / 20)
    return active + noise


def run_sweep(percentiles=PERCENTILES) -> Sweep:
    duty, snr, value = [], [], []
    for d in DUTIES:
        for s in TRUE_SNRS_DB:
            for seed in SEEDS:
                wav = synthesise(d, s, np.random.default_rng(hash((d, s, seed)) % 2**32))
                duty.append(d)
                snr.append(s)
                value.append(statistic(wav, SAMPLE_RATE, percentiles))
    return Sweep(np.array(duty), np.array(snr), np.array(value))


def part_limiting_case() -> None:
    heading(7, "The limiting case, on synthetic signals")
    sweep = run_sweep()
    print("  " + " " * 8 + "".join(f"{s:8.0f}" for s in TRUE_SNRS_DB) + "      span")
    for d in DUTIES:
        row = [sweep.value[(sweep.duty == d) & (sweep.snr == s)].mean() for s in TRUE_SNRS_DB]
        print(f"  {d:6.2f}  " + "".join(f"{v:8.1f}" for v in row) + f"{max(row) - min(row):10.1f}")

    full_range = TRUE_SNRS_DB[-1] - TRUE_SNRS_DB[0]
    blind = [d for d in DUTIES if sweep.span_at(d) < 2.0]
    print(f"\n  a true estimate would show a span of {full_range:.0f} dB at every duty cycle")
    if blind:
        print(f"  the statistic is blind at duty cycles {', '.join(f'{d:.2f}' for d in blind)}")
        print("  A source with no intrinsic modulation has no dispersion to report, whatever its")
        print("  ratio. Real continuous sources modulate, so part 5 shows a weak response rather")
        print("  than none. The mechanism is the same; only its degree differs.")
    else:
        print("  the statistic tracks the imposed ratio at every duty cycle tested")


# --------------------------------------------------------------------------- entry point

def main() -> None:
    print("Audit of the frame-percentile statistic used as a rejection axis")
    print("=" * 74)
    part_identity()
    part_gain()
    part_silence()
    part_floor()
    part_imposed_noise()
    part_categories()
    part_limiting_case()


if __name__ == "__main__":
    main()
