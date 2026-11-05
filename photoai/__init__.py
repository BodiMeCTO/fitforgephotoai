"""fitforgephotoai: a synthetic dataset of garments draped on bodies, each sample carrying the
exact ease (garment girth minus body girth, per point) and the fabric's elasticity it was made
with, plus renders of it. Built to test whether a photo model can learn ease and elasticity.

The solver (``fitforge``) and the body model (``annyext``) are reached ONLY through
``photoai.engine``; nothing else in this package imports them.
"""
__version__ = "0.1.0"
