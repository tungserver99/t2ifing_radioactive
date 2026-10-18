"""Bounded inference batching and optional data residency; never changes training batches."""
import torch

EVENTS = []
BATCH_LIMITS = {}
GIB = 2 ** 30


def image01(x, name="image"):
    """Validate RGB at image boundaries without silently casting or clipping."""
    if x.dtype != torch.float32 or x.ndim != 4 or x.shape[1] != 3 or not x.numel():
        raise ValueError(f"{name}: expected nonempty FP32 NCHW RGB")
    with torch.no_grad():
        if not bool(torch.isfinite(x).all() & (x >= 0).all() & (x <= 1).all()):
            raise ValueError(f"{name}: expected finite values in [0,1]")
    return x


def decoded01(raw):
    # Check before clipping: Inf must not become an apparently valid pixel.
    if raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()):
        raise ValueError("VAE output must be finite FP32")
    return image01((raw / 2 + .5).clamp(0, 1), "decoded image")


def batch_size(requested, device, cap=16):
    if requested < 0:
        raise ValueError("Batch size must be >=0 (0 = auto)")
    if requested:
        return requested
    if torch.device(device).type != "cuda":
        return 1
    # Start optimistically and let batches() halve on a real allocation failure.
    # A static bytes/item estimate selected batch=1 on a shared H200 even though
    # VAE-only stages could process much larger batches.
    return cap


def batches(items, fn, size, stage="inference"):
    """Commit each batch only after success. fn must not write files or mutate models.

    Generation callbacks recreate generators from per-image seeds on each retry.
    Finite checks and all other errors remain fatal; only allocation failures retry.
    """
    if size < 1:
        raise ValueError("Batch size must be positive")
    size = min(size, BATCH_LIMITS.get(stage, size))
    start = 0
    while start < len(items):
        chunk = items[start:start + size]
        retry = False
        try:
            result = fn(chunk)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            size = max(1, len(chunk) // 2)
            BATCH_LIMITS[stage] = size
            retry = True
        if retry:
            # Outside except: release failed forward's traceback/activation references.
            torch.cuda.empty_cache()
            EVENTS.append({"stage": stage, "event": "oom_backoff", "batch_size": size})
            print(f"{stage}: retrying with batch_size={size}", flush=True)
            continue
        yield start, result
        start += len(chunk)


class DataCache:
    def __init__(self, device, mode="auto", max_gib=16., reserve_fraction=.5):
        if mode not in ("auto", "cpu") or max_gib < 0 or not 0 < reserve_fraction < 1:
            raise ValueError("Invalid data-cache budget")
        self.device = torch.device(device)
        self.mode, self.max_bytes = mode, int(max_gib * GIB)
        self.reserve_fraction, self.used = reserve_fraction, 0
        self.stats = []

    def promote(self, tensors, label):
        """Keep CPU ownership with caller; return GPU copies only within both budgets."""
        needed = sum(t.numel() * t.element_size() for t in tensors)
        allowed = False
        if self.mode == "auto" and self.device.type == "cuda" and needed <= self.max_bytes - self.used:
            free, total = torch.cuda.mem_get_info(self.device)
            allowed = free - needed >= max(2 * GIB, total * self.reserve_fraction)
        target = "cpu"
        if allowed:
            moved = []
            try:
                for tensor in tensors:
                    moved.append(tensor.to(self.device))
                self.used += needed
                target = str(self.device)
            except torch.cuda.OutOfMemoryError:
                moved.clear()
            if target != "cpu":
                self.stats.append({"label": label, "bytes": needed, "device": target})
                return moved
            torch.cuda.empty_cache()
        self.stats.append({"label": label, "bytes": needed, "device": target})
        return tensors


def all_finite(tensors):
    """One host synchronization for all parameter/gradient finite checks."""
    values = list(tensors)
    return bool(torch.stack([torch.isfinite(t).all() for t in values]).all()) if values else True
