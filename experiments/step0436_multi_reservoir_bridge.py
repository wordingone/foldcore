"""
Step 436 — DEFERRED, never ran.

Spec (fc0de35d:L242): "Is cross-domain survival bidirectional?
Step 433: P-MNIST->LS20->P-MNIST = 0.0pp contamination.
Reverse: Run LS20 FIRST (10K steps), then P-MNIST (10 tasks).
Measure: LS20 unique states, P-MNIST accuracy, codebook sizes."

After Step 435 (EWC/replay comparison), Leo sent Step 437 spec
(minimal self-modifying reservoir) and the session pivoted to the
reservoir series (437-465). Step 436 was never placed back in queue.

The cross-domain survival series closed with Step 439 (mirror-side
conclusion: reservoir reflects back to codebook). Step 436 (reverse
direction) was rendered less urgent and permanently deferred.
"""
# Deferred — bidirectional survival test never run. Series closed at step 439.
