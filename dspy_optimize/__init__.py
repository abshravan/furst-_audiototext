"""DSPy-based prompt optimization wrapper around MOSS-Audio inference.

This package sits on top of `infer.py` without modifying it. The
optimization target is the *instruction text* (and optionally few-shot
demos) sent to MOSS-Audio for binary frustration classification.
"""
