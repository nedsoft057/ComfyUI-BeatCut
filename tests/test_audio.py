import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))); sys.path.insert(0, os.path.dirname(__file__))
from synth import synth_track
from beatcut.audio import analyze
import numpy as np
def test_tempo_and_drop():
    for bpm in (100.0, 128.0, 140.0):
        w, sr, gt = synth_track(bpm=bpm)
        bm = analyze(w, sr)
        print(bpm, bm.summary())
        assert abs(bm.tempo - bpm) / bpm < 0.04, (bm.tempo, bpm)
        spb = 60 / bpm
        # beats land on the grid
        err = [min(abs((b - gt["first_beat"]) / spb - round((b - gt["first_beat"]) / spb)) for _ in [0]) for b in bm.beats]
        assert np.median(err) < 0.12, np.median(err)
        assert bm.drops, "no drop found"
        assert min(abs(d - gt["drop_time"]) for d in bm.drops) < 0.15, (bm.drops, gt["drop_time"])
if __name__ == "__main__":
    test_tempo_and_drop(); print("OK")
