"""Training orchestrator (placeholder)."""


def main():
    """Run distributed training (TODO: implement)."""
    # Planned architecture:
    # - Spawn N actor processes (CPU)
    # - Spawn 1 learner process (GPU/MPS)
    # - Share replay buffer via multiprocessing
    # - Periodically sync weights from learner to actors
    pass
