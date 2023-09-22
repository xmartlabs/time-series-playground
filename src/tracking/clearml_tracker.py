from clearml import Task, Logger, OutputModel
from .tracker import Tracker


class ClearMLTracker(Tracker):

    def __init__(self, project_name=None, experiment_name=None):
        self.task = Task.current_task() or Task.init(project_name=project_name, task_name=experiment_name,
                                                     auto_connect_frameworks={'tensorboard': True, 'matplotlib': True}
                                                     )
        self.logger = Logger.current_logger()
        self._callback = None

    def execute_remotely(self, queue_name):
        self.task.execute_remotely(queue_name=queue_name)

    def track_config(self, config):
        self.task.set_parameters_as_dict(config)

    def track_artifacts(self, filepath, name=None):
        self.task.upload_artifact(name, artifact_object=filepath)

    def track_model(self, filepath, name=None):
        output_model = OutputModel(task=self.task, framework="TensorFlow")
        output_model.tags = [name]
        output_model.update_weights_package(weights_path=filepath, auto_delete_file=False)

    def log_scalar_metric(self, metric, series, iteration, value):
        if iteration is None:
            self.logger.report_single_value(metric, value)
        else:
            self.logger.report_scalar(metric, series, iteration=iteration, value=value)

    def log_chart(self, title, series, iteration, figure):
        self.task.logger.report_matplotlib_figure(title=title, series=series, iteration=iteration, figure=figure)

    def add_tags(self, tags):
        self.task.add_tags(tags)

    def finish_run(self):
        self.task.mark_completed()
        self.task.close()

    # def get_callback(self):
    #     if self._callback is None:
    #         from src.tracking.keras_tracking_callback import ClearMLTrainTrackingCallback
    #         self._callback = ClearMLTrainTrackingCallback(self.logger)
    #     return self._callback
