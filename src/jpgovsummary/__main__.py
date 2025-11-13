"""
Entry point for jpgovsummary package when run as a module.
"""

import sys

from .jpgovwatcher import main

if __name__ == "__main__":
    sys.exit(main())
