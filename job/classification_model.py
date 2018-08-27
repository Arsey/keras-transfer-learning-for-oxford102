from __future__ import absolute_import

from .job import Job
from easyclassify.utils import subclass, override


@subclass
class ClassificationModelJob(Job):

    def __init__(self, dataset_id, **kwargs):
        super(ClassificationModelJob, self).__init__(**kwargs)
        self.dataset_id = dataset_id

    @override
    def job_type(self):
        return 'Image Classification Model'

    # @override
    # def json_dict(self, verbose=False):
    #     d = super(ModelJob, self).json_dict(verbose)
    #     d['dataset_id'] = self.dataset_id
    #
    #     if verbose:
    #         d.update({
    #             'snapshots': [s[1] for s in self.train_task().snapshots],
    #         })
    #     return d

    def train_task(self):
        """Return the first TrainTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.TrainTask)][0]
