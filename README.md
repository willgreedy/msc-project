# Dendritic Error Network

An implementation of the multi-compartment dendritic error network from [Sacramento et al.](https://arxiv.org/pdf/1810.11393.pdf)

# Requirements

* Python (3.6.8)
* NumPy (1.16.4)
* MatPlotLib (2.0.2)
* PyTorch (1.1.0; Optional)

# Usage
```
python experiment_builder.py [-monitored_values_config_file MONITORED_VALUES_CONFIG_FILE]
                             [-self_predict_phase_length SELF_PREDICT_PHASE_LENGTH]
                             [-test_phase_length TEST_PHASE_LENGTH]
                             [-example_iterations EXAMPLE_ITERATIONS]
                             [-target_network_weights_path TARGET_NETWORK_WEIGHTS_PATH]
                             [-model_file MODEL_FILE]
                             [-train_data_path TRAIN_DATA_PATH]
                             [-test_data_path TEST_DATA_PATH]
                             [-resume_from_epoch RESUME_FROM_EPOCH]
                             [-show_final_plots SHOW_FINAL_PLOTS]
                             [-parameter_config_file PARAMETER_CONFIG_FILE]
                             experiment_name num_epochs num_epoch_iterations
```
