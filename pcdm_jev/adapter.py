"""jevbench adapter for PCDM, mirroring jevbench.adapters.local_openjev (in-process, no
network, no tariff -> price null). `--model` = run dir, `--endpoint` = energy run dir (compose).
Needs the jevbench clone on sys.path (scripts/jevbench_run.py does that)."""
import json
import time

from jevbench.adapters.base import DecisionResult, build_question

from .decider import PCDMDecider


class LocalPCDMAdapter:
    name = "local_pcdm"
    cost_basis = "local_gpu_no_provider_tariff"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, revision=None,
                 mode="energy", device="auto", max_state=4096, backbone=None, tap_layer=0, shots=0,
                 prompt_style="ours"):
        self.path = model  # run dir (None for mode="mcq_zero_shot": frozen backbone, no checkpoint)
        self.model = (f"pcdm:{mode}:{model or backbone}" + (f":shots{shots}" if shots else "")
                      + (f":{prompt_style}" if prompt_style != "ours" else ""))
        self.energy_run = endpoint
        self.key_env, self.timeout_s, self.revision = key_env, timeout_s, revision
        self.price_input_per_m, self.price_output_per_m = price_input_per_m, price_output_per_m
        self.mode, self.device, self.max_state = mode, device, max_state
        self.backbone, self.tap_layer, self.shots, self.prompt_style = backbone, tap_layer, shots, prompt_style
        self._decider = None

    def load(self):
        if self._decider is None:
            t0 = time.perf_counter()
            self._decider = PCDMDecider(self.path, device=self.device, mode=self.mode,
                                        energy_run=self.energy_run, max_state=self.max_state,
                                        backbone=self.backbone, tap_layer=self.tap_layer, shots=self.shots,
                                        prompt_style=self.prompt_style)
            # ponytail: one throwaway decision so CUDA init lands in load_s, not item 1's latency
            self._decider.decide("x", {"type": "noul", "instructions": "?"}, ["yes", "no"])
            self.load_s = time.perf_counter() - t0
        return self._decider

    def build_request(self, task) -> dict:
        state = task.state if isinstance(task.state, str) else json.dumps(task.state, ensure_ascii=False)
        return {"state": state, "question": build_question(task), "labels": list(task.labels)}

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        body = self.build_request(task)
        res.request_body = {k: v for k, v in body.items() if k != "labels"}
        try:
            decider = self.load()
        except Exception as e:  # noqa: BLE001 - a failed load is a failed attempt
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res
        t0 = time.perf_counter()
        try:
            probs, runtime = decider.decide(body["state"], body["question"], body["labels"])
        except Exception as e:  # noqa: BLE001
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.latency_s = time.perf_counter() - t0
        res.probs, res.raw, res.ok = probs, {"runtime": runtime}, True
        # ponytail: suffix_tokens doesn't exist yet (no native path reports it) -- .get keeps this correct if it's added later
        res.usage = {"input_tokens": runtime.get("state_tokens", 0) + runtime.get("query_tokens", 0)
                     + runtime.get("suffix_tokens", 0), "output_tokens": 0}
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
