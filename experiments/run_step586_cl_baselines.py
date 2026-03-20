"""
Step 586 — Continual Learning baselines: CIFAR-100 split into 5 tasks.

Reviewer gap: no comparisons to standard CL methods.
Implements three baselines on sequential 5-task CIFAR-100:
  A) Naive fine-tuning (no replay, no regularization) — catastrophic forgetting lower bound
  B) Replay buffer (50 examples per class, 10% replay ratio) — practical upper bound
  C) EWC (Elastic Weight Consolidation, Kirkpatrick 2017) — regularization baseline

Each method trains on Task 1..5 sequentially, then evaluates on ALL tasks.
Reports per-task accuracy and average accuracy after all tasks seen.

Compare: our substrate (mgu + 581d) vs standard CL methods.
Note: substrate comparison requires a separate ARC-game run (out of scope here).
This script establishes the CIFAR-100 baseline numbers only.

Runtime: ~15-30 min on GPU (A100). Run via train daemon.
"""
import time
import numpy as np
import sys

# ── Config ────────────────────────────────────────────────────────────────────

N_TASKS = 5
CLASSES_PER_TASK = 20       # 100 CIFAR classes / 5 tasks
EPOCHS_PER_TASK = 10
BATCH_SIZE = 128
LR = 1e-3
REPLAY_BUFFER_SIZE = 50     # examples per class
REPLAY_RATIO = 0.1          # 10% of each batch from replay buffer
EWC_LAMBDA = 5000           # EWC regularization strength (Kirkpatrick et al. default range)
SEED = 42


# ── Model ─────────────────────────────────────────────────────────────────────

def make_model(num_classes=100):
    """Small ResNet-like CNN for CIFAR-100."""
    import torch.nn as nn
    return nn.Sequential(
        # Block 1
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        # Block 2
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        # Block 3
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        # Head
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


# ── Data ──────────────────────────────────────────────────────────────────────

def get_cifar100_tasks(seed=SEED):
    """Split CIFAR-100 into N_TASKS sequential tasks."""
    import torch
    from torchvision import datasets, transforms

    rng = np.random.RandomState(seed)
    class_order = rng.permutation(100).tolist()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_full = datasets.CIFAR100(root='/tmp/cifar100', train=True,
                                    download=True, transform=transform_train)
    test_full = datasets.CIFAR100(root='/tmp/cifar100', train=False,
                                   download=True, transform=transform_test)

    tasks_train = []
    tasks_test = []
    for t in range(N_TASKS):
        task_classes = class_order[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK]
        task_classes_set = set(task_classes)
        class_remap = {c: i for i, c in enumerate(task_classes)}

        # Filter train
        train_idx = [i for i, (_, c) in enumerate(train_full) if c in task_classes_set]
        train_data = [(train_full[i][0], class_remap[train_full[i][1]]) for i in train_idx]

        # Filter test
        test_idx = [i for i, (_, c) in enumerate(test_full) if c in task_classes_set]
        test_data = [(test_full[i][0], class_remap[test_full[i][1]]) for i in test_idx]

        tasks_train.append((task_classes, train_data))
        tasks_test.append((task_classes, test_data))

    return tasks_train, tasks_test


class TaskDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Training utilities ────────────────────────────────────────────────────────

def make_loader(data, batch_size=BATCH_SIZE, shuffle=True):
    import torch
    from torch.utils.data import DataLoader
    ds = TaskDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=lambda b: (
                          torch.stack([x for x, y in b]),
                          torch.tensor([y for x, y in b])
                      ))


def train_epoch(model, loader, optimizer, criterion, device,
                replay_loader=None, task_offset=0):
    import torch
    model.train()
    total_loss = 0; correct = 0; total = 0

    for x, y in loader:
        x, y = x.to(device), (y + task_offset).to(device)

        if replay_loader is not None:
            try:
                rx, ry = next(replay_iter)
            except (NameError, StopIteration):
                replay_iter = iter(replay_loader)
                rx, ry = next(replay_iter)
            # Select replay subset (REPLAY_RATIO of current batch size)
            n_replay = max(1, int(len(x) * REPLAY_RATIO))
            rx, ry = rx[:n_replay].to(device), ry[:n_replay].to(device)
            x = torch.cat([x, rx])
            y = torch.cat([y, ry])

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += len(x)

    return total_loss / total, correct / total


def evaluate(model, loader, device, task_offset=0):
    import torch
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), (y + task_offset).to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(x)
    return correct / total if total > 0 else 0.0


# ── EWC ───────────────────────────────────────────────────────────────────────

class EWC:
    """Elastic Weight Consolidation (Kirkpatrick et al. 2017)."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.fisher = {}    # param name -> Fisher information (diagonal)
        self.optima = {}    # param name -> optimal params after task

    def compute_fisher(self, loader, task_offset=0, n_samples=200):
        """Estimate diagonal Fisher on task data after training."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        total = 0

        for x, y in loader:
            if total >= n_samples:
                break
            x, y = x.to(self.device), (y + task_offset).to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            log_probs = F.log_softmax(out, dim=1)
            # Sample from model's own distribution
            targets = torch.multinomial(log_probs.exp(), 1).squeeze()
            loss = F.nll_loss(log_probs, targets)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            total += len(x)

        for n in fisher:
            fisher[n] /= total
            # Accumulate: new Fisher + old Fisher (multi-task)
            self.fisher[n] = fisher[n] + self.fisher.get(n, torch.zeros_like(fisher[n]))
            self.optima[n] = self.model.state_dict()[n].detach().clone()

    def penalty(self):
        """EWC regularization loss term."""
        import torch
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optima[n]) ** 2).sum()
        return EWC_LAMBDA * loss


# ── Method runners ────────────────────────────────────────────────────────────

def run_naive(tasks_train, tasks_test, device):
    """Naive sequential fine-tuning — no replay, no regularization."""
    import torch
    import torch.nn as nn

    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss()
    task_accs = []  # per-task accuracy after all tasks seen

    # Track per-task test loaders with offsets
    test_loaders = []
    for t, (task_classes, test_data) in enumerate(tasks_test):
        test_loaders.append((t, make_loader(test_data, shuffle=False), t * CLASSES_PER_TASK))

    for t, (task_classes, train_data) in enumerate(tasks_train):
        task_offset = t * CLASSES_PER_TASK
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        train_loader = make_loader(train_data)
        t0 = time.time()

        for epoch in range(EPOCHS_PER_TASK):
            loss, acc = train_epoch(model, train_loader, optimizer, criterion, device,
                                    task_offset=task_offset)
            if epoch == EPOCHS_PER_TASK - 1:
                print(f"  Task {t+1} ep{epoch+1}: loss={loss:.3f} acc={acc:.3f} {time.time()-t0:.0f}s",
                      flush=True)

    # Evaluate on all tasks
    accs = []
    for t, loader, offset in test_loaders:
        acc = evaluate(model, loader, device, task_offset=offset)
        accs.append(acc)
        print(f"  Task {t+1} test acc: {acc:.3f}", flush=True)
    print(f"  Average acc (naive): {np.mean(accs):.3f}", flush=True)
    return accs


def run_replay(tasks_train, tasks_test, device):
    """Replay buffer: store REPLAY_BUFFER_SIZE examples per class, mix into future batches."""
    import torch
    import torch.nn as nn
    import random

    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss()
    replay_bank = []   # global replay store (x, global_y)

    test_loaders = []
    for t, (task_classes, test_data) in enumerate(tasks_test):
        test_loaders.append((t, make_loader(test_data, shuffle=False), t * CLASSES_PER_TASK))

    for t, (task_classes, train_data) in enumerate(tasks_train):
        task_offset = t * CLASSES_PER_TASK
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        train_loader = make_loader(train_data)

        # Build replay loader from bank
        replay_loader = None
        if replay_bank:
            replay_data = random.sample(replay_bank,
                                        min(len(replay_bank), BATCH_SIZE * EPOCHS_PER_TASK))
            replay_loader = make_loader(replay_data, batch_size=max(1, BATCH_SIZE // 4))

        t0 = time.time()
        for epoch in range(EPOCHS_PER_TASK):
            loss, acc = train_epoch(model, train_loader, optimizer, criterion, device,
                                    replay_loader=replay_loader, task_offset=task_offset)
            if epoch == EPOCHS_PER_TASK - 1:
                print(f"  Task {t+1} ep{epoch+1}: loss={loss:.3f} acc={acc:.3f} {time.time()-t0:.0f}s",
                      flush=True)

        # Add current task examples to replay bank
        rng = np.random.RandomState(t)
        class_examples = {}
        for x, y in train_data:
            c = y + task_offset
            class_examples.setdefault(c, []).append((x, c))
        for c, examples in class_examples.items():
            selected = rng.choice(len(examples),
                                  min(REPLAY_BUFFER_SIZE, len(examples)),
                                  replace=False)
            replay_bank.extend([examples[i] for i in selected])

    # Evaluate
    accs = []
    for t, loader, offset in test_loaders:
        acc = evaluate(model, loader, device, task_offset=offset)
        accs.append(acc)
        print(f"  Task {t+1} test acc: {acc:.3f}", flush=True)
    print(f"  Average acc (replay): {np.mean(accs):.3f}", flush=True)
    return accs


def run_ewc(tasks_train, tasks_test, device):
    """EWC: elastic weight consolidation after each task."""
    import torch
    import torch.nn as nn

    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss()
    ewc = EWC(model, device)

    test_loaders = []
    for t, (task_classes, test_data) in enumerate(tasks_test):
        test_loaders.append((t, make_loader(test_data, shuffle=False), t * CLASSES_PER_TASK))

    for t, (task_classes, train_data) in enumerate(tasks_train):
        task_offset = t * CLASSES_PER_TASK
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        train_loader = make_loader(train_data)
        t0 = time.time()

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            total_loss = 0; correct = 0; total_n = 0
            for x, y in train_loader:
                x, y = x.to(device), (y + task_offset).to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                if t > 0:
                    loss = loss + ewc.penalty()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_n += len(x)
            if epoch == EPOCHS_PER_TASK - 1:
                print(f"  Task {t+1} ep{epoch+1}: loss={total_loss/total_n:.3f} "
                      f"acc={correct/total_n:.3f} {time.time()-t0:.0f}s", flush=True)

        # Compute Fisher for EWC consolidation
        if t < N_TASKS - 1:
            ewc.compute_fisher(train_loader, task_offset=task_offset)

    # Evaluate
    accs = []
    for t, loader, offset in test_loaders:
        acc = evaluate(model, loader, device, task_offset=offset)
        accs.append(acc)
        print(f"  Task {t+1} test acc: {acc:.3f}", flush=True)
    print(f"  Average acc (EWC): {np.mean(accs):.3f}", flush=True)
    return accs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Step 586: CL baselines on CIFAR-100 (5 tasks, {CLASSES_PER_TASK} classes each)",
          flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Config: epochs={EPOCHS_PER_TASK} batch={BATCH_SIZE} lr={LR}", flush=True)
    print(f"  Replay buffer: {REPLAY_BUFFER_SIZE}/class  EWC lambda: {EWC_LAMBDA}", flush=True)

    print("\nLoading CIFAR-100...", flush=True)
    tasks_train, tasks_test = get_cifar100_tasks()
    print(f"  {N_TASKS} tasks, {CLASSES_PER_TASK} classes each", flush=True)
    print(f"  Task sizes: {[len(td) for _, td in tasks_train]} train, "
          f"{[len(td) for _, td in tasks_test]} test", flush=True)

    results = {}

    print("\n=== Method A: Naive fine-tuning ===", flush=True)
    t0 = time.time()
    results['naive'] = run_naive(tasks_train, tasks_test, device)
    print(f"  Naive elapsed: {time.time()-t0:.0f}s", flush=True)

    print("\n=== Method B: Replay buffer ===", flush=True)
    t0 = time.time()
    results['replay'] = run_replay(tasks_train, tasks_test, device)
    print(f"  Replay elapsed: {time.time()-t0:.0f}s", flush=True)

    print("\n=== Method C: EWC ===", flush=True)
    t0 = time.time()
    results['ewc'] = run_ewc(tasks_train, tasks_test, device)
    print(f"  EWC elapsed: {time.time()-t0:.0f}s", flush=True)

    # Summary table
    print(f"\n{'='*60}")
    print(f"Step 586: CL Baselines — CIFAR-100 Split-5")
    print(f"  {'Method':<12} | {'T1':>6} {'T2':>6} {'T3':>6} {'T4':>6} {'T5':>6} | {'Avg':>6}")
    print(f"  {'-'*60}")
    for method, accs in results.items():
        padded = accs + [0.0] * (N_TASKS - len(accs))
        row = '  '.join(f"{a:.3f}" for a in padded[:N_TASKS])
        avg = np.mean(accs) if accs else 0.0
        print(f"  {method:<12} | {row} | {avg:.3f}")

    # Forgetting metric (T1 acc after all tasks vs T1 acc after task 1 only)
    print(f"\n  Forgetting on Task 1 (final T1 acc):")
    for method, accs in results.items():
        if accs:
            print(f"    {method}: {accs[0]:.3f}")

    print(f"\nNOTE: Compare substrate (mgu + 581d) ARC-game results to these CL baselines")
    print(f"in the paper. Different domains — use average accuracy and forgetting metrics.")


if __name__ == "__main__":
    main()
