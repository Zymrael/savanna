.PHONY: quality style

check_dirs := savanna
# Check that source code meets quality standards

# this target runs checks on all files
check:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	ruff check --fix $(check_dirs)
	ruff format $(check_dirs)
