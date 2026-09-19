from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BOPO_TSP = Path("/workspace/groupopt-modern-hosts/BOPO/TSP")
OFFICIAL_REPRO = Path("/workspace/groupopt-bopo-h2/official-repro")
CHECKPOINT = BOPO_TSP / "result/saved_n50/checkpoint-200.pt"
sys.path.insert(0, str(BOPO_TSP))
from TSPModel import TSPModel  # noqa: E402

MODEL_PARAMS = {
    "start_node": "pomo",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def model_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        t = value.detach().cpu().contiguous()
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def gather_nodes(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return values.gather(1, index[:, :, None].expand(index.size(0), index.size(1), values.size(-1)))

def reshape_heads(values: torch.Tensor, heads: int) -> torch.Tensor:
    b, n, _ = values.shape
    return values.reshape(b, n, heads, -1).transpose(1, 2)

def safe_entropy(log_p: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(log_p)
    safe = torch.where(finite, log_p, torch.zeros_like(log_p))
    p = safe.exp() * finite.to(log_p.dtype)
    return -(p * safe).sum(-1)

def path_state_features(
    enc: torch.Tensor, pred: torch.Tensor, comp: torch.Tensor
) -> torch.Tensor:
    r, n, d = enc.shape
    ci = comp.unsqueeze(-1)
    ei = ci.expand_as(enc)
    comp_sum_by = torch.zeros_like(enc).scatter_add(1, ei, enc)
    comp_sum = comp_sum_by.gather(1, ei)
    ones = torch.ones_like(ci, dtype=enc.dtype)
    comp_size_by = torch.zeros_like(ones).scatter_add(1, ci, ones)
    comp_size = comp_size_by.gather(1, ci)
    comp_mean = comp_sum / comp_size
    is_start = pred.lt(0)
    start_source = enc * is_start.unsqueeze(-1)
    start_by = torch.zeros_like(enc).scatter_add(1, ei, start_source)
    start = start_by.gather(1, ei)
    return torch.cat((comp_mean, start, comp_size / float(n)), dim=-1)

def packed_open_components(
    succ: torch.Tensor, pred: torch.Tensor, comp: torch.Tensor
):
    r, n = succ.shape
    m = int((succ[0] < 0).sum().item())
    if not torch.all((succ < 0).sum(1).eq(m)):
        raise RuntimeError("rollouts disagree on number of open tails")
    if not torch.all((pred < 0).sum(1).eq(m)):
        raise RuntimeError("rollouts disagree on number of open heads")
    tails = (succ < 0).nonzero(as_tuple=False)[:, 1].reshape(r, m)
    heads = (pred < 0).nonzero(as_tuple=False)[:, 1].reshape(r, m)
    ct = comp.gather(1, tails)
    ch = comp.gather(1, heads)
    same = ct[:, :, None].eq(ch[:, None, :])
    if not torch.all(same.sum(-1).eq(1)):
        raise RuntimeError("each open tail must have exactly one path start")
    start_pos = same.to(torch.long).argmax(-1)
    starts = heads.gather(1, start_pos)
    return tails, heads, starts, same

@torch.no_grad()
def bopo_all_source_head_proposal(
    model: TSPModel,
    enc: torch.Tensor,
    coords: torch.Tensor,
    succ: torch.Tensor,
    pred: torch.Tensor,
    comp: torch.Tensor,
    edges_added: int,
):
    r, n, d = enc.shape
    tails, heads, starts, same = packed_open_components(succ, pred, comp)
    m = tails.size(1)
    tail_e = gather_nodes(enc, tails)
    head_e = gather_nodes(enc, heads)
    start_e = gather_nodes(enc, starts)
    dec = model.decoder
    h = MODEL_PARAMS["head_num"]
    q = reshape_heads(dec.Wq_first(start_e), h) + reshape_heads(dec.Wq_last(tail_e), h)
    k = reshape_heads(dec.Wk(head_e), h)
    v = reshape_heads(dec.Wv(head_e), h)
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))
    pair_mask = same if edges_added < n - 1 else torch.zeros_like(same)
    if pair_mask.all(-1).any():
        raise RuntimeError("nonterminal BOPO proposal row has no legal endpoint")
    score = score.masked_fill(pair_mask[:, None], -torch.inf)
    weights = torch.softmax(score, dim=-1)
    attended = torch.matmul(weights, v)
    concat = attended.transpose(1, 2).reshape(r, m, -1)
    mh = dec.multi_head_combine(concat)
    logits = torch.matmul(mh, head_e.transpose(1, 2))
    logits = MODEL_PARAMS["logit_clipping"] * torch.tanh(logits / MODEL_PARAMS["sqrt_embedding_dim"])
    logits = logits.masked_fill(pair_mask, -torch.inf)
    log_p = torch.log_softmax(logits, dim=-1)

    tail_xy = gather_nodes(coords, tails)
    head_xy = gather_nodes(coords, heads)
    distances = (tail_xy[:, :, None, :] - head_xy[:, None, :, :]).norm(p=2, dim=-1)
    finite = torch.isfinite(log_p)
    probs = torch.where(finite, log_p.exp(), torch.zeros_like(log_p))
    safe = torch.where(finite, log_p, torch.zeros_like(log_p))
    entropy = -(probs * safe).sum(-1) / math.log(max(n, 2))
    expected_cost = (probs * distances).sum(-1) / math.sqrt(2.0)
    best_prob = probs.amax(-1)
    summary = torch.stack((entropy, expected_cost, best_prob), dim=-1)
    return tails, heads, log_p, summary

class FullASCCAdapter(nn.Module):
    def __init__(self, d: int = 128, heads: int = 8, qkv: int = 16, clip: float = 10.0):
        super().__init__()
        self.d = d
        self.heads = heads
        self.clip = clip
        self.project_tail_state = nn.Linear(2 * d + 1, d, bias=False)
        self.project_head_summary = nn.Linear(3, d, bias=False)
        self.Wq_first = nn.Linear(d, heads * qkv, bias=False)
        self.Wq_last = nn.Linear(d, heads * qkv, bias=False)
        self.Wk = nn.Linear(d, heads * qkv, bias=False)
        self.Wv = nn.Linear(d, heads * qkv, bias=False)
        self.multi_head_combine = nn.Linear(heads * qkv, d)

    def tail_log_probabilities(
        self,
        anchor_e: torch.Tensor,
        last_head_e: torch.Tensor,
        enc: torch.Tensor,
        path_features: torch.Tensor,
        proposal_summary_full: torch.Tensor,
        tail_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidates = enc + self.project_tail_state(path_features) + self.project_head_summary(proposal_summary_full)
        q = reshape_heads(self.Wq_first(anchor_e[:, None]), self.heads)
        q = q + reshape_heads(self.Wq_last(last_head_e[:, None]), self.heads)
        k = reshape_heads(self.Wk(candidates), self.heads)
        v = reshape_heads(self.Wv(candidates), self.heads)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(tail_mask[:, None, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        concat = attended.transpose(1, 2).reshape(enc.size(0), 1, -1)
        glimpse = self.multi_head_combine(concat).squeeze(1)
        logits = torch.matmul(glimpse[:, None], candidates.transpose(1, 2)).squeeze(1)
        logits = self.clip * torch.tanh(logits / math.sqrt(self.d))
        logits = logits.masked_fill(tail_mask, -torch.inf)
        return torch.log_softmax(logits, dim=-1)

def expand_rollouts(coords: torch.Tensor, enc: torch.Tensor, k: int):
    b, n, _ = coords.shape
    d = enc.size(-1)
    coords_r = coords[:, None].expand(b, k, n, 2).reshape(b * k, n, 2).contiguous()
    enc_r = enc[:, None].expand(b, k, n, d).reshape(b * k, n, d).contiguous()
    anchors = torch.arange(k, device=coords.device, dtype=torch.long)[None].expand(b, k).reshape(-1)
    anchor_e = enc_r[torch.arange(b * k, device=coords.device), anchors]
    return coords_r, enc_r, anchors, anchor_e

def rollout_full_ascc(
    adapter: FullASCCAdapter,
    bopo: TSPModel,
    coords: torch.Tensor,
    enc: torch.Tensor,
    k: int,
    sample_source: bool,
    action_generator: torch.Generator | None = None,
):
    coords, enc, anchors, anchor_e = expand_rollouts(coords, enc, k)
    r, n, _ = coords.shape
    device = coords.device
    batch = torch.arange(r, device=device)
    succ = torch.full((r, n), -1, dtype=torch.long, device=device)
    pred = torch.full((r, n), -1, dtype=torch.long, device=device)
    comp = torch.arange(n, device=device, dtype=torch.long)[None].expand(r, n).clone()
    last_head_e = anchor_e
    total_cost = torch.zeros(r, device=device)
    source_log_likelihood = torch.zeros(r, device=device)
    entropy_sum = torch.zeros(r, device=device)

    for edge in range(n):
        tails, heads, packed_head_log_p, packed_summary = bopo_all_source_head_proposal(
            bopo, enc, coords, succ, pred, comp, edge
        )
        summary_full = torch.zeros((r, n, 3), device=device, dtype=enc.dtype)
        summary_full.scatter_(1, tails[:, :, None].expand(r, tails.size(1), 3), packed_summary)
        pstate = path_state_features(enc, pred, comp)
        tail_mask = succ.ge(0)

        if edge == 0:
            selected_tail = anchors
            selected_tail_log_p = torch.zeros(r, device=device)
            tail_entropy = torch.zeros(r, device=device)
        else:
            tail_log_p = adapter.tail_log_probabilities(
                anchor_e, last_head_e, enc, pstate, summary_full, tail_mask
            )
            tail_entropy = safe_entropy(tail_log_p)
            if sample_source:
                probs = tail_log_p.exp()
                selected_tail = torch.multinomial(probs, 1, generator=action_generator).squeeze(1)
            else:
                selected_tail = tail_log_p.argmax(-1)
            selected_tail_log_p = tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)

        source_pos = tails.eq(selected_tail[:, None]).to(torch.long).argmax(1)
        selected_head_log_p = packed_head_log_p[batch, source_pos]
        selected_head_pos = selected_head_log_p.argmax(-1)
        selected_head = heads[batch, selected_head_pos]

        tail_xy = coords[batch, selected_tail]
        head_xy = coords[batch, selected_head]
        total_cost = total_cost + (tail_xy - head_xy).norm(p=2, dim=-1)
        source_log_likelihood = source_log_likelihood + selected_tail_log_p
        entropy_sum = entropy_sum + tail_entropy

        succ = succ.clone()
        pred = pred.clone()
        succ[batch, selected_tail] = selected_head
        pred[batch, selected_head] = selected_tail
        if edge < n - 1:
            left = comp.gather(1, selected_tail[:, None])
            right = comp.gather(1, selected_head[:, None])
            merged = torch.minimum(left, right)
            in_merge = comp.eq(left) | comp.eq(right)
            comp = torch.where(in_merge, merged, comp)
        last_head_e = enc[batch, selected_head]

    if not succ.ge(0).all() or not pred.ge(0).all():
        raise RuntimeError("Full ASCC did not produce a cycle cover")
    return total_cost, source_log_likelihood, entropy_sum / float(n)

@torch.no_grad()
def encode_frozen(bopo: TSPModel, coords: torch.Tensor) -> torch.Tensor:
    return bopo.encoder(coords)

@torch.no_grad()
def evaluate_random(
    adapter: FullASCCAdapter,
    bopo: TSPModel,
    coords: torch.Tensor,
    k: int,
) -> float:
    adapter.eval()
    enc = encode_frozen(bopo, coords)
    costs, _, _ = rollout_full_ascc(adapter, bopo, coords, enc, k, False)
    return float(costs.reshape(coords.size(0), k).amin(1).double().mean().item())

@torch.no_grad()
def evaluate_official(
    adapter: FullASCCAdapter,
    bopo: TSPModel,
    chunk_augmented: int,
):
    adapter.eval()
    parts = []
    started = time.monotonic()
    for step in range(0, 1000, 10):
        problems, _ = torch.load(
            BOPO_TSP / f"data/tsp_n50_{step}.pkl",
            map_location="cuda",
            weights_only=False,
        )
        chunks = []
        for a in range(0, problems.size(0), chunk_augmented):
            c = problems[a:a + chunk_augmented]
            enc = encode_frozen(bopo, c)
            costs, _, _ = rollout_full_ascc(adapter, bopo, c, enc, 50, False)
            chunks.append(costs.reshape(c.size(0), 50).cpu())
        batch_costs = torch.cat(chunks, dim=0).reshape(8, 10, 50)
        parts.append(batch_costs)
    raw = torch.cat(parts, dim=1)
    best = raw.amin(2).amin(0)
    return raw, best, time.monotonic() - started

def paired_ci(diff: torch.Tensor, samples: int = 5000):
    x = diff.double().cpu()
    g = torch.Generator(device="cpu").manual_seed(20260918)
    means = []
    for _ in range(0, samples, 250):
        m = min(250, samples - len(means) * 250)
        idx = torch.randint(0, x.numel(), (m, x.numel()), generator=g)
        means.append(x[idx].mean(1))
    y = torch.cat(means)
    q = torch.quantile(y, torch.tensor([0.025, 0.975], dtype=torch.double))
    return [float(q[0]), float(q[1])]

def load_bopo(device: torch.device):
    model = TSPModel(**MODEL_PARAMS).to(device)
    ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def assert_frozen(bopo: TSPModel):
    if any(p.requires_grad for p in bopo.parameters()):
        raise RuntimeError("BOPO parameter unexpectedly trainable")
    if any(p.grad is not None for p in bopo.parameters()):
        raise RuntimeError("BOPO parameter unexpectedly has gradient")

def save_adapter(path: Path, adapter: FullASCCAdapter, step: int, val: float, config: dict):
    tmp = path.with_suffix(".tmp")
    torch.save({"adapter": adapter.state_dict(), "step": step, "validation": val, "config": config}, tmp)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--validation-size", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--pomo-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--final-chunk", type=int, default=20)
    ap.add_argument("--skip-final", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.jsonl"
    config = vars(args).copy()
    config["out"] = str(config["out"])
    (args.out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    ck_sha_before = file_sha256(CHECKPOINT)
    bopo = load_bopo(device)
    bopo_hash_before = model_sha256(bopo)
    frozen_snapshot = {k: v.detach().cpu().clone() for k, v in bopo.state_dict().items()}
    assert_frozen(bopo)

    adapter = FullASCCAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    adapter_params = sum(p.numel() for p in adapter.parameters())

    val_g = torch.Generator(device="cpu").manual_seed(2026091801)
    validation = torch.rand((args.validation_size, 50, 2), generator=val_g).to(device)
    action_g = torch.Generator(device=device).manual_seed(args.seed + 2)
    data_g = torch.Generator(device=device).manual_seed(args.seed + 1)

    initial_val = evaluate_random(adapter, bopo, validation, args.pomo_size)
    best_val = initial_val
    best_step = 0
    save_adapter(args.out / "best.pt", adapter, 0, best_val, config)
    print(f"initial_validation={initial_val:.9f} adapter_params={adapter_params}", flush=True)

    started = time.monotonic()
    for step in range(1, args.steps + 1):
        adapter.train()
        coords = torch.rand((args.batch_size, 50, 2), generator=data_g, device=device)
        enc = encode_frozen(bopo, coords)
        costs, source_ll, tail_entropy = rollout_full_ascc(
            adapter, bopo, coords, enc, args.pomo_size, True, action_g
        )
        grouped = costs.reshape(args.batch_size, args.pomo_size)
        grouped_ll = source_ll.reshape(args.batch_size, args.pomo_size)
        advantage = (grouped - grouped.mean(1, keepdim=True)).detach()
        loss = (advantage * grouped_ll).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
        assert_frozen(bopo)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate_random(adapter, bopo, validation, args.pomo_size)
            if val < best_val:
                best_val, best_step = val, step
                save_adapter(args.out / "best.pt", adapter, step, val, config)
            metric = {
                "step": step,
                "train_cost_mean": float(costs.detach().double().mean()),
                "train_best_pomo_mean": float(grouped.detach().amin(1).double().mean()),
                "policy_loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "tail_entropy": float(tail_entropy.detach().double().mean()),
                "validation_best_pomo_mean": val,
                "best_validation": best_val,
                "best_step": best_step,
                "elapsed_seconds": time.monotonic() - started,
                "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(metric, sort_keys=True) + "\n")
            print(json.dumps(metric, sort_keys=True), flush=True)

        if step % 500 == 0 or step == args.steps:
            save_adapter(args.out / "latest.pt", adapter, step, best_val, config)

    payload = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    adapter.load_state_dict(payload["adapter"], strict=True)
    best_step = int(payload["step"])
    best_val = float(payload["validation"])

    summary = {
        "round_id": "E4-H2-BOPO-FULL-ASCC",
        "mechanism": "full_construction_order_path_state_plus_all_source_bopo_endpoint_proposals_plus_last_head_attention",
        "host": "BOPO-2025",
        "host_upstream_commit": "2739fbfe39a478755173c892b80c4b06f9f59b05",
        "host_checkpoint": "checkpoint-200.pt",
        "host_checkpoint_sha256_before": ck_sha_before,
        "bopo_state_sha256_before": bopo_hash_before,
        "bopo_trainable_parameters": 0,
        "adapter_parameters": adapter_params,
        "training_steps": args.steps,
        "training_batch_size": args.batch_size,
        "training_pomo_size": args.pomo_size,
        "learning_rate": args.lr,
        "best_step": best_step,
        "best_validation_best_pomo_mean": best_val,
        "initial_validation_best_pomo_mean": initial_val,
        "official_test_used_for_training_or_selection": False,
        "endpoint_policy_frozen": True,
        "encoder_frozen": True,
    }

    if not args.skip_final:
        raw, full_best, eval_seconds = evaluate_official(adapter, bopo, args.final_chunk)
        official = torch.load(
            OFFICIAL_REPRO / "official_instance_aug_best_costs.pt",
            map_location="cpu",
            weights_only=False,
        )["costs"].double()
        full_best_d = full_best.double()
        diff = official - full_best_d
        torch.save({"costs": raw}, args.out / "full_ascc_all_aug_pomo_costs.pt")
        torch.save({"costs": full_best}, args.out / "full_ascc_instance_aug_best_costs.pt")
        torch.save({"improvement": diff}, args.out / "paired_improvement.pt")
        eps = 1e-9
        summary.update({
            "official_instances": 1000,
            "official_augmentations": 8,
            "official_pomo": 50,
            "matched_control_mean": float(official.mean()),
            "full_ascc_mean": float(full_best_d.mean()),
            "absolute_mean_improvement": float(diff.mean()),
            "relative_improvement_percent": float(diff.mean() / official.mean() * 100.0),
            "paired_improvement_ci95": paired_ci(diff),
            "improved_instances": int((diff > eps).sum()),
            "worse_instances": int((diff < -eps).sum()),
            "ties": int((diff.abs() <= eps).sum()),
            "median_improvement": float(diff.median()),
            "eval_seconds": eval_seconds,
        })

    ck_sha_after = file_sha256(CHECKPOINT)
    bopo_hash_after = model_sha256(bopo)
    exact_equal = all(torch.equal(v.detach().cpu(), frozen_snapshot[k]) for k, v in bopo.state_dict().items())
    assert_frozen(bopo)
    summary["host_checkpoint_sha256_after"] = ck_sha_after
    summary["bopo_state_sha256_after"] = bopo_hash_after
    summary["checkpoint_file_unchanged"] = ck_sha_before == ck_sha_after
    summary["bopo_state_bitwise_unchanged"] = exact_equal and bopo_hash_before == bopo_hash_after
    if not summary["checkpoint_file_unchanged"] or not summary["bopo_state_bitwise_unchanged"]:
        raise RuntimeError("BOPO checkpoint/state changed during ASCC training")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("=== FULL ASCC SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()

