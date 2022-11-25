import os
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

dependencies = [
  "os",
  "pdoc3",
  "pkg_resources"
]

# Documentation https://towardsdatascience.com/how-to-generate-professional-api-docs-in-minutes-from-docstrings-aed0341bbda7
pkg_resources.require(dependencies)

path_to_lib = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "survivors"))
print(path_to_lib)
os.system(f"pdoc --http localhost:8080 -c latex_math=True {path_to_lib}")
