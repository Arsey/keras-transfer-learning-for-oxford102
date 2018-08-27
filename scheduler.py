from __future__ import absolute_import

from collections import OrderedDict
import gevent
import gevent.event
from job.job import Job, JOBS_DIR
import os
import shutil
import signal
from status import Status
import traceback
import time
from .log import logger

NON_PERSISTENT_JOB_DELETE_TIMEOUT_SECONDS = 3600


class Resource(object):
    """
    Stores information about which tasks are using a resource
    """

    class ResourceAllocation(object):
        """
        Marks that a task is using [part of] a resource
        """

        def __init__(self, task, value):
            """
            Arguments:
            task -- which task is using the resource
            value -- how much of the resource is being used
            """
            self.task = task
            self.value = value

    def __init__(self, identifier=None, max_value=1):
        """
        Keyword arguments:
        identifier -- some way to identify this resource
        max_value -- a numeric representation of the capacity of this resource
        """
        if identifier is None:
            self.identifier = id(self)
        else:
            self.identifier = identifier
        self.max_value = max_value
        self.allocations = []

    def remaining(self):
        """
        Returns the amount of this resource that is not being used
        """
        return self.max_value - sum(a.value for a in self.allocations)

    def allocate(self, task, value):
        """
        A task is requesting to use this resource
        """
        if self.remaining() - value < 0:
            raise RuntimeError('Resource is already maxed out at %s/%s' % (self.remaining(), self.max_value))
        self.allocations.append(self.ResourceAllocation(task, value))

    def deallocate(self, task):
        """
        The task has finished using this resource
        """
        for i, a in enumerate(self.allocations):
            if id(task) == id(a.task):
                self.allocations.pop(i)
                return True
        return False


class Scheduler:

    def __init__(self, gpu_list=None, verbose=True):
        """
        Keyword arguments:
        gpu_list -- a comma-separated string which is a list of GPU id's
        """
        self.jobs = OrderedDict()
        self.running = False
        self.shutdown = gevent.event.Event()
        self.verbose = verbose

        # Keeps track of resource usage
        self.resources = {
            # TODO: break this into CPU cores, memory usage, IO usage, etc.
            'parse_folder_task_pool': [Resource()],
            'create_db_task_pool': [Resource(max_value=4)],
            'analyze_db_task_pool': [Resource(max_value=4)],
            'inference_task_pool': [Resource(max_value=4)],
            'gpus': [Resource(identifier=index) for index in gpu_list.split(',')] if gpu_list else [],
        }

    def load_past_jobs(self):
        """
        Look in the jobs directory and load all valid jobs
        """
        loaded_jobs = []
        failed_jobs = []
        for dir_name in sorted(os.listdir(JOBS_DIR)):
            if os.path.isdir(os.path.join(JOBS_DIR, dir_name)):
                # Make sure it hasn't already been loaded
                if dir_name in self.jobs:
                    continue

                try:
                    job = Job.load(dir_name)
                    # The server might have crashed
                    if job.status.is_running():
                        job.status = Status.ABORT
                    for task in job.tasks:
                        if task.status.is_running():
                            task.status = Status.ABORT

                    # We might have changed some attributes here or in __setstate__
                    job.save()
                    loaded_jobs.append(job)
                except Exception as e:
                    failed_jobs.append((dir_name, e))

        # add ModelJobs
        for job in loaded_jobs:
            self.jobs[job.id()] = job

        logger.info('Loaded %d jobs.' % len(self.jobs))

        if len(failed_jobs):
            logger.warning('Failed to load %d jobs.' % len(failed_jobs))
            if self.verbose:
                for job_id, e in failed_jobs:
                    logger.debug('%s - %s: %s' % (job_id, type(e).__name__, str(e)))

    def add_job(self, job):
        """
        Add a job to self.jobs
        """
        if not self.running:
            logger.error('Scheduler not running. Cannot add job.')
            return False
        else:
            self.jobs[job.id()] = job

            # Need to fix this properly
            # if True or flask._app_ctx_stack.top is not None:
            from easyclassify.app import app, socketio
            with app.app_context():
                # send message to job_management room that the job is added

                socketio.emit('job update',
                              {
                                  'update': 'added',
                                  'job_id': job.id(),
                              },
                              namespace='/jobs',
                              room='job_management',
                              )

            return True

    def start(self):
        if self.running:
            return True

        gevent.spawn(self.main_thread)

        self.running = True
        return True

    def stop(self):
        """
        Stop the Scheduler
        Returns True if the shutdown was graceful
        """
        self.shutdown.set()
        wait_limit = 5
        start = time.time()
        while self.running:
            if time.time() - start > wait_limit:
                return False
            time.sleep(0.1)
        return True

    def main_thread(self):
        """
        Monitors the jobs in current_jobs, updates their statuses,
        and puts their tasks in queues to be processed by other threads
        """
        signal.signal(signal.SIGTERM, self.sigterm_handler)
        try:
            last_saved = None
            while not self.shutdown.is_set():
                # Iterate backwards so we can delete jobs
                for job in self.jobs.values():
                    if job.status == Status.INIT:
                        def start_this_job(job):
                            job.status = Status.RUN

                        # Delay start by one second for initial page load
                        gevent.spawn_later(1, start_this_job, job)

                    if job.status == Status.WAIT:
                        job.status = Status.RUN

                    if job.status == Status.RUN:
                        alldone = True
                        for task in job.tasks:
                            if task.status in [Status.INIT, Status.WAIT]:
                                alldone = False
                                # try to start the task
                                if task.ready_to_queue():
                                    requested_resources = task.offer_resources(self.resources)
                                    if requested_resources is None:
                                        task.status = Status.WAIT
                                    else:
                                        if self.reserve_resources(task, requested_resources):
                                            gevent.spawn(self.run_task,
                                                         task, requested_resources)
                            elif task.status == Status.RUN:
                                # job is not done
                                alldone = False
                            elif task.status in [Status.DONE, Status.ABORT]:
                                # job is done
                                pass
                            elif task.status == Status.ERROR:
                                # propagate error status up to job
                                job.status = Status.ERROR
                                alldone = False
                                break
                            else:
                                logger.warning('Unrecognized task status: "%s"', task.status, job_id=job.id())
                        if alldone:
                            job.status = Status.DONE
                            logger.info('Job complete.', job_id=job.id())

                            job.save()

                # save running jobs every 15 seconds
                if not last_saved or time.time() - last_saved > 15:
                    for job in self.jobs.values():
                        if job.status.is_running():
                            if job.is_persistent():
                                job.save()
                        elif (not job.is_persistent() and
                              (time.time() - job.status_history[-1][1] >
                               NON_PERSISTENT_JOB_DELETE_TIMEOUT_SECONDS)):
                            # job has been unclaimed for far too long => proceed to garbage collection
                            self.delete_job(job)
                    last_saved = time.time()

                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

        # Shutdown
        for job in self.jobs.values():
            job.abort()
            job.save()
        self.running = False

    def reserve_resources(self, task, resources):
        """
        Reserve resources for a task
        """
        try:
            # reserve resources
            for resource_type, requests in resources.iteritems():
                for identifier, value in requests:
                    found = False
                    for resource in self.resources[resource_type]:
                        if resource.identifier == identifier:
                            resource.allocate(task, value)
                            self.emit_gpus_available()
                            found = True
                            break
                    if not found:
                        raise RuntimeError('Resource "%s" with identifier="%s" not found' % (
                            resource_type, identifier))
            task.current_resources = resources
            return True
        except Exception as e:
            self.task_error(task, e)
            self.release_resources(task, resources)
            return False

    def release_resources(self, task, resources):
        """
        Release resources previously reserved for a task
        """
        # release resources
        for resource_type, requests in resources.iteritems():
            for identifier, value in requests:
                for resource in self.resources[resource_type]:
                    if resource.identifier == identifier:
                        resource.deallocate(task)
                        self.emit_gpus_available()
        task.current_resources = None

    def task_error(self, task, error):
        """
        Handle an error while executing a task
        """
        print('{}: {}'.format(type(error).__name__, error), 'job_id', task.job_id)
        task.exception = error
        task.traceback = traceback.format_exc()
        print(task.traceback)
        task.status = Status.ERROR

    def run_task(self, task, resources):
        """
        Executes a task

        Arguments:
        task -- the task to run
        resources -- the resources allocated for this task
            a dict mapping resource_type to lists of (identifier, value) tuples
        """
        try:
            task.run(resources)
        except Exception as e:
            self.task_error(task, e)
        finally:
            self.release_resources(task, resources)

    def sigterm_handler(self, signal, frame):
        self.shutdown.set()

    def get_job(self, job_id):
        """
        Look through self.jobs to try to find the Job
        Returns None if not found
        """
        if job_id is None:
            return None
        return self.jobs.get(job_id, None)

    def delete_job(self, job):
        """
        Deletes an entire job folder from disk
        Returns True if the Job was found and deleted
        """
        if isinstance(job, str) or isinstance(job, unicode):
            job_id = job
        elif isinstance(job, Job):
            job_id = job.id()
        else:
            raise ValueError('called delete_job with a %s' % type(job))

        # try to find the job
        job = self.jobs.get(job_id, None)
        if job:
            self.jobs.pop(job_id, None)
            job.abort()
            if os.path.exists(job.dir()):
                shutil.rmtree(job.dir())
            print('Job deleted', job_id)

            # TODO: add socketio emit event that job has been deleted

            return True

        # see if the folder exists on disk
        path = os.path.join(JOBS_DIR, job_id)
        if os.path.dirname(path) == JOBS_DIR and os.path.exists(path):
            shutil.rmtree(path)
            return True

        return False

    # TODO: implement if needed
    def emit_gpus_available(self):
        print('emit gpus available')
