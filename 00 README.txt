COMPLETE PROJECT CODE AND DATA DIRECTORY
	The directory contains each and every files created for the preprocessing, training, testing and inference.
Navigation:
	These are the important locations to quickly navigate through the most relevant items.
	1. Current Directory: Used to test out best models by pre-processing and splitting logically and train several models. Each script performs a specific function. The final training 	code is 05_train_model_modular.py, which is a modular code which can take in various model architectures as plug in. The inputs and outputs remaining similar.
	2. DEPLOYMENT_TRAINING:  Contains the deployment training, where the whole data is used to train the best model found from the 1. folder. Contains documentation README to clarify.
	3. FINAL_INFERENCE_PIPELINE: Contains the final inference workflow, where unseen data arrives, is preprocessed for inference and goes through the trained classification model.
	4. Stats n insights: Contains useful stats and insights saved during preprocessing and model selection procedure.