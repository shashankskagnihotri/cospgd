"""
Agnihotri, Shashank, Jung, Steffen, Keuper, Margret. "CosPGD: a unified white-box adversarial attack for pixel-wise prediction tasks." 
arXiv preprint arXiv:2302.02213 (2023).

A tool for benchmarking adversarial robustness of pixel-wise prediction tasks.
"""

__version__ = "0.1.0"
__author__ = 'Shashank Agnihotri, Steffen Jung, Prof. Dr. Margret Keuper'
__credits__ = 'Keuper Labs'


from cospgd.attack_implementations import Attack as Attack
functions = Attack()