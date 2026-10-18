# Provenance and third-party notices

This repository selectively exports the Fusion Research Agent application and
the minimal local TORAX/TD3/SAC shadow runner from the hfmarl-fusion working tree.
It retains the original Apache-2.0 notices. Consult flower-app/MERGE.md for the
Fusion Investigator/THERMAL merge history and original component attribution.
The export manifest records the copied source byte hashes; copied files may
include work that had not yet been committed in the original repository.

The 3D HTML exports load Three.js 0.128.0, OrbitControls and STLLoader from pinned
CDNs. Three.js is MIT licensed: https://github.com/mrdoob/three.js/blob/r128/LICENSE.
These visual assets are illustrations, not calibrated reactor geometry.

TORAX, JAX, PyTorch, Flower, LangGraph and other installed dependencies remain
subject to their respective licenses. Package source is not vendored here.
The app's use of the OpenAI Python client is solely an API-compatible client for
the configured Nebius Token Factory endpoint in the research workflow.

Evidence bundles contain only the explicitly selected terminal-run records and
their exact frozen source archives. They are measurements from a synthetic
SPARC-like circular TORAX model, not data from an operating reactor. Web sources
are retrieved at runtime through Tavily and are not republished in this export.
