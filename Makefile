.DEFAULT_GOAL := default
#### ACTIONS ####

reinstall_package:
	@pip uninstall -y deepdive || :
	@pip install -e .

download_data:
	python -c 'from deepdive.ml_logic.data import download_data; download_data()'

load_data:
	python -c 'from deepdive.ml_logic.data import load_data; load_data()'
