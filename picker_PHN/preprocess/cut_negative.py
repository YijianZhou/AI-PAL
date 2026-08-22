"""Disabled in positive-only CEED workflow.

This folder trains only on known earthquake windows. Negative sample cutting is
intentionally not used here.
"""

if __name__ == '__main__':
    raise SystemExit('cut_negative.py is disabled for positive-only CEED training')
