"""Post-hoc bug fix, evaluated once on the test: STD-k on near-constant channels.

The bug: STD-k divides by each channel's raw training std (+1e-8); the near-constant-channel
floor (config sd_floor, 1e-3) was applied to the model inputs only. A channel flat during
training gets std ~1e-8, so any later change alarms for as long as it lasts.
The fix: STD-k is not applied to channels whose training std is below sd_floor (it has no
spread to measure against); those channels keep Telemanom and silence. Same threshold as the
model's floor, not tuned. Model, errors and the frozen detector setting are unchanged.

  python -m core.evaluate_fix --mission M1

Writes to out_full/fix_stdfloor/<mk>/ (the pre-registered result.json is left untouched).
"""
import os
import shutil
import sys

import numpy as np

from . import config
from . import period as PR
from . import evaluate as E
from . import evaluate_par as EP

_init = PR.Period.__init__


def _patched(self, *a, **k):
    _init(self, *a, **k)
    floor = config.MISSIONS[self.mk]["sd_floor"]
    flat = self.vsd < floor
    self.vsd = np.where(flat, np.inf, self.vsd).astype(np.float32)     # |v - mu| / inf = 0
    if self.name == "test":
        print(f"[{self.mk}] STD-k disabled on {int(flat.sum())} near-constant channels "
              f"(train std < {floor:g}): {np.flatnonzero(flat).tolist()}", flush=True)


def main():
    mk = sys.argv[sys.argv.index("--mission") + 1]
    dst = f"{config.OUT}/fix_stdfloor"
    os.makedirs(f"{dst}/{mk}", exist_ok=True)
    shutil.copy(f"{config.OUT}/{mk}/frozen.json", f"{dst}/{mk}/frozen.json")
    PR.Period.__init__ = _patched
    EP.OUT = E.OUT = dst                    # outputs only; models / errors still from job_dir
    EP.main()


if __name__ == "__main__":
    main()
