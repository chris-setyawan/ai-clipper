"""
Run the pipeline from a clone, without installing anything.

    python run_pipeline.py <video> <transcript.json> -o out --clips 15

The real implementation lives in the package (ai_clipper.cli.pipeline) so that
an installed copy has it too, where it is called `ai-clipper`. This file only
puts src/ on the path first, for the case where nothing has been installed yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_clipper.cli.pipeline import main

if __name__ == "__main__":
    main()
