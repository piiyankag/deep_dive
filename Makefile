.DEFAULT_GOAL := default
#### ACTIONS ####

reinstall_package:
	@pip uninstall -y deepdive || :
	@pip install -e .

download_data:
	python -c 'from deepdive.ml_logic.data import download_data; download_data()'

load_data:
	python -c 'from deepdive.ml_logic.data import load_data; load_data()'

load_model:
	python -c 'from deepdive.ml_logic.registry import load_model; load_model()'

get_classes:
	python -c 'from deepdive.ml_logic.data import get_class_names; get_class_names()'

run_train:
	python -c 'from deepdive.interface.main import train; train()'

run_detailed_evaluation:
	python -c 'from deepdive.interface.main import detailed_evaluation; detailed_evaluation()'

run_api:
	uvicorn deepdive.api.fast:app --reload

reset_local_files:
	rm -rf ${ML_DIR}
	echo "Preparing local files in ${HOME_PATH}"
	mkdir -p ${HOME_PATH}/lewagon/deep_dive/data/
	mkdir ${HOME_PATH}/lewagon/deep_dive/data/raw
	mkdir ${HOME_PATH}/lewagon/deep_dive/data/processed
	mkdir ${HOME_PATH}/lewagon/deep_dive/training_outputs
	mkdir ${HOME_PATH}/lewagon/deep_dive/training_outputs/metrics
	mkdir ${HOME_PATH}/lewagon/deep_dive/training_outputs/models
	mkdir ${HOME_PATH}/lewagon/deep_dive/training_outputs/params
