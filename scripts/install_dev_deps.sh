#!/usr/bin/env bash
set -e

DEPS=$(python <<EOF
import re
import sys

with open("setup.py", "r") as f:
    content = f.read()

match = re.search(r"dev_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
if not match:
    sys.exit("Could not find dev_requires in setup.py")

# Each line in dev_requires is of the form "something>=version"
deps_raw = match.group(1)
deps_list = [
    dep.strip().strip("\"'") for dep in deps_raw.split(",")
    if dep.strip()
]
print(" ".join(deps_list))
EOF
)

echo "Installing dev requirements:"
echo "${DEPS}"

# Install
pip install --upgrade pip
pip install ${DEPS}
