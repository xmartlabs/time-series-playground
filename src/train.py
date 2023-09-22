import argparse
import yaml
import tensorflow as tf

from training.window_generator import WindowGenerator
from training.configs.train_config import TrainConfig
from training.configs.model_config import ModelConfig, ModelType
from preprocessing.jena_preprocessing import JenaPreprocessor
from tracking.clearml_tracker import ClearMLTracker


def main(train_config: TrainConfig, model_config: ModelConfig, tracker: ClearMLTracker):
    jp = JenaPreprocessor()
    jp.preprocess()

    tracker.add_tags(["single_step" if train_config.shift == 1 else "multi_step"])
    window = WindowGenerator(input_width=train_config.input_width,
                             label_width=train_config.label_width,
                             shift=train_config.shift,
                             label_columns=['T (degC)'],
                             train_df=jp.train_df,
                             val_df=jp.val_df,
                             test_df=jp.test_df)

    if model_config.nn_model_type == ModelType.baseline:
        model = model_config.nn_model_type.model(label_index=window.column_indices['T (degC)'])
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
    else:
        model = model_config.nn_model_type.model()
        model.build_model()
        history = model.compile_and_fit(window, epochs=train_config.epochs)
        print(history)

    val_score = model.model.evaluate(window.val)
    test_score = model.model.evaluate(window.test, verbose=0)
    window.plot(tracker, model.model)
    print(f'Validation MAE: {val_score[1]:.2f}, Test MAE: {test_score[1]:.2f}')


def load_config(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-e', '--experiment', type=str, required=True)
    args = parser.parse_args()

    tracker = ClearMLTracker(project_name='Time Series PG', experiment_name=args.experiment)
    config = load_config(args.config)
    tracker.track_config(config)
    train_config = TrainConfig(**config['train'])
    model_config = ModelConfig(**config['model'])
    main(train_config, model_config, tracker)
    tracker.finish_run()
