#!/bin/sh
#
# Pre-commit hook for clearing output cells from commited analysis Jupyter notebooks.
#

echo "Running pre-commit hook to clear output from deliver/*.ipynb notebooks."
for notebook in git diff --cached --name-only -- 'deliver/*.ipynb'
do
	echo "Clearing output from $notebook"
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearOutputPreprocessor.remove_metadata_fields=[] --to notebook --inplace $notebook
	git add $notebook
done