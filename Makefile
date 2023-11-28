.DEFAULT_GOAL := default
#### ACTIONS ####

reinstall_package:
	@pip uninstall -y deepdive || :
	@pip install -e .

download_data:
	python -c 'from deepdive.ml_logic.data import download_data; download_data()'

load_data:
	python -c 'from deepdive.ml_logic.data import load_data; load_data()'

get_classes:
	python -c 'from deepdive.ml_logic.data import get_class_names; get_class_names()'

run_train:
	python -c 'from deepdive.interface.main import train; train()'

run_detailed_evaluation:
	python -c 'from deepdive.interface.main import detailed_evaluation; detailed_evaluation()'

run_api:
	uvicorn deepdive.api.fast:app --reload
