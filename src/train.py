import argparse
import os
import tensorflow as tf
import tempfile
import yaml

from training.window_generator import WindowGenerator
from training.configs.train_config import TrainConfig
from training.configs.model_config import ModelConfig, ModelType
from preprocessing.jena_preprocessing import JenaPreprocessor
from preprocessing.power_preprocessing import PowerPreprocessor
from tracking.clearml_tracker import ClearMLTracker


def main(train_config: TrainConfig, model_config: ModelConfig, tracker: ClearMLTracker, dataset: str):
    assert dataset in ['jena', 'power'], "Invalid dataset"
    target_col = 'T (degC)' if dataset == 'jena' else 'total'
    include_context = model_config.nn_model_type == ModelType.ctxt_conv_rnn

    pp = JenaPreprocessor() if dataset == 'jena' else PowerPreprocessor()
    pp.preprocess(include_context=include_context)

    tracker.add_tags(["single_step" if train_config.shift == 1 else "multi_step",
                      dataset])
    window = WindowGenerator(input_width=train_config.input_width,
                             label_width=train_config.label_width,
                             shift=train_config.shift,
                             label_columns=[target_col],
                             train_df=pp.train_df,
                             val_df=pp.val_df,
                             test_df=pp.test_df,
                             include_context=include_context,
                             sequence_stride=train_config.sequence_stride)

    if model_config.nn_model_type == ModelType.baseline:
        label_index = window.column_indices[target_col] if train_config.shift == 1 else None
        model = model_config.nn_model_type.model(label_index=label_index)
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
    else:
        model = model_config.nn_model_type.model()
        model.build_model(**model_config.model_params(), input_width=train_config.input_width,)
        model.compile_and_fit(window, lr=train_config.learning_rate, epochs=train_config.epochs)

    train_score, val_score, test_score = model.get_eval_scores(window)
    if model_config.nn_model_type == ModelType.xgboost:
        model.plot(tracker)
    else:
        window.plot(tracker, model, plot_col=target_col)
    print(f'Validation MAE: {val_score:.2f}, Test MAE: {test_score:.2f}')
    tracker.log_scalar_metric("Train MAE", None, None, train_score)
    tracker.log_scalar_metric("Val. MAE", None, None, val_score)
    tracker.log_scalar_metric("Test MAE", None, None, test_score)

    output_folder = tempfile.mkdtemp()
    output_file_name = os.path.join(output_folder, 'output_model')
    model.save_model(output_file_name)


def load_config(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-e', '--experiment', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['jena', 'power'])
    args = parser.parse_args()

    tracker = ClearMLTracker(project_name='Time Series PG', experiment_name=args.experiment)
    config = load_config(args.config)
    tracker.track_config(config)
    train_config = TrainConfig(**config['train'])
    model_config = ModelConfig.from_config(config)
    main(train_config, model_config, tracker, args.dataset)
    tracker.finish_run()
