"""Development only: with SOUP_SOLUTION_DEBUG=1 in the environment, print GPU memory per step.
Never set in measured runs; without the variable nothing here runs."""

import os
import time


def install():
    if os.environ.get("SOUP_SOLUTION_DEBUG") != "1":
        return
    import torch
    import transformers

    class MemoryLog(transformers.TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            print(f"[mem] train begin: allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB, "
                  f"peak so far {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
            torch.cuda.reset_peak_memory_stats()

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
            self.a0 = torch.cuda.memory_allocated()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            from . import unpad
            print(f"[mem] step {state.global_step}: {time.perf_counter() - self.t0:.3f} s, "
                  f"start {self.a0 / 2**20:.0f} MiB, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, "
                  f"tokens {getattr(unpad, 'LAST_TOKENS', None)}", flush=True)
            if state.global_step <= 2:
                from . import warmup
                t = dict(warmup.TIMES)
                if "start" in t and "done" in t:
                    t["compile_s"] = round(t["done"] - t["start"], 2)
                print(f"[mem] warm-up: {t}", flush=True)
            torch.cuda.reset_peak_memory_stats()

    original = transformers.Trainer.__init__

    def init(self, *a, **k):
        original(self, *a, **k)
        self.add_callback(MemoryLog())

    transformers.Trainer.__init__ = init
