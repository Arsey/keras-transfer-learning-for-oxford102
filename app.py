import flask
from flask import render_template
import config
import util
import time
import numpy as np
import io
import traceback
import argparse
from keras.applications.imagenet_utils import preprocess_input
import os
import dataset
from scheduler import Scheduler
from job import ClassificationModelJob
from task.train import TrainTask
from flask_socketio import SocketIO
from gevent import monkey
import utils.time_filters

monkey.patch_all()

app = flask.Flask(__name__)
app.secret_key = os.urandom(12).encode('hex')
app.config['WTF_CSRF_ENABLED'] = False
socketio = SocketIO(app, async_mode='gevent', path='/socket.io')

app.jinja_env.filters['print_time'] = utils.time_filters.print_time
app.jinja_env.filters['print_time_diff'] = utils.time_filters.print_time_diff
app.jinja_env.filters['print_time_since'] = utils.time_filters.print_time_since
app.jinja_env.filters['sizeof_fmt'] = utils.sizeof_fmt

scheduler = Scheduler(gpu_list='1,2')
scheduler.load_past_jobs()
# TODO: implement
# scheduler.load_past_jobs()

model = {}
model_module = {}
novelty_detection_clf = {}
activation_function = {}


def init(model_name):
    util.set_img_format()
    util.tf_allow_growth()
    config.model = model_name


def load_model():
    model_module = util.get_model_class_instance()
    model = model_module.load()
    print('Model loaded')

    print('Warming up the model')
    tic = time.time()
    input_shape = (1,) + model_module.img_size + (3,)
    dummy_img = np.ones(input_shape)
    dummy_img = preprocess_input(dummy_img)
    model.predict(dummy_img)
    print('Warming up took {} s'.format(time.time() - tic))

    # # save compiled model to wait less next time
    # # TODO: consider on saving it also at the train stage
    # json_string = model.to_json()
    # open('trained/model-compiled-resnet50.json', 'w').write(json_string)

    return model_module, model


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


def get_request_images():
    image = flask.request.files.get('image')
    if image:
        return [image]
    else:
        return flask.request.files.getlist('image[]')


def load_request_images(images):
    loaded_images = []
    images_names = []
    for img in images:
        images_names.append(img.filename)
        img = model_module.load_img(io.BytesIO(img.read()))
        loaded_images.append(img)

    return np.array(loaded_images), images_names


class PredictionInterpreter(object):
    def __init__(self):
        class_indices = dict(zip(config.classes, range(len(config.classes))))
        self.keys = list(class_indices.keys())
        self.values = list(class_indices.values())

    def get_label(self, index):
        return self.keys[self.values[index]]


@app.route("/predict", methods=["POST"])
def predict():
    batch_size = flask.request.form.get('batch_size', default=1, type=int)

    data = {
        'success': False
    }

    images = get_request_images()
    if images:
        try:
            images, images_names = load_request_images(images)
            out = model.predict(images, batch_size=batch_size, verbose=1)

            data['results'] = {}
            for i, img_name in enumerate(images_names):
                img_results = out[i]
                sorted_out = img_results.argsort()[::-1]
                data['results'][img_name] = []
                for t in sorted_out:
                    r = {
                        "label": prediction_interpreter.get_label(t),
                        "probability": float(img_results[t])
                    }
                    data['results'][img_name].append(r)
            data['success'] = True

        except Exception as e:
            print('Error', e)
            data['error'] = e
            traceback.print_stack()

            data = {
                'success': False
            }

    return flask.jsonify(data)


@app.route('/datasets/delete/<name>', methods=['GET'])
def datasets_delete(name):
    dataset.delete(name)
    return flask.redirect(flask.url_for('datasets'))


@app.route('/datasets', methods=['POST', 'GET'])
def datasets():
    form = dataset.NewDatasetFrom()
    datasets = dataset.get_datasets_list()
    if form.validate_on_submit():
        dataset.create(
            train_folder=form.train_folder.data,
            dataset_name=form.dataset_name.data,
            resize_channels=form.resize_channels.data,
            resize_width=form.resize_width.data,
            resize_height=form.resize_height.data,
            resize_mode=form.resize_mode.data,
            encoding=form.encoding.data,
            folder_train_min_per_class=form.folder_train_min_per_class.data,
            folder_train_max_per_class=form.folder_train_max_per_class.data,
            folder_pct_val=form.folder_pct_val.data,
            folder_pct_test=form.folder_pct_test.data,
        )
    return render_template('datasets.html', form=form, datasets=datasets)


@app.route('/train', methods=['POST', 'GET'])
def train():
    job = None
    dataset_id = 'cor_clusters'
    try:
        job = ClassificationModelJob(
            # TODO get name from form data
            name='test',
            # TODO get dataset from form data
            dataset_id=dataset_id
        )
        task = TrainTask(
            job=job,
            dataset=dataset_id,
            train_epochs=1,
            snapshot_interval=1,
            learning_rate=1e-5,
            lr_policy='fixed',
            gpu_count=1,
            # selected_gpus=selected_gpus,
            batch_size=32,
            # batch_accumulation=form.batch_accumulation.data,
            # val_interval=form.val_interval.data,
            # traces_interval=form.traces_interval.data,
            # pretrained_model=pretrained_model,
            # crop_size=form.crop_size.data,
            # use_mean=form.use_mean.data,
            network='resnet50',
            # random_seed=form.random_seed.data,
            # solver_type=form.solver_type.data,
            # rms_decay=form.rms_decay.data,
            # shuffle=form.shuffle.data,
            # data_aug=data_aug,
        )
        job.tasks.append(task)
        scheduler.add_job(job)

    except Exception as e:
        # traceback.print_stack()
        # traceback.print_exc()
        # print(e)
        if job:
            scheduler.delete_job(job)
        raise

    return render_template('train.html')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, required=True, help='Base model architecture', choices=[
#         config.MODEL_RESNET50,
#         config.MODEL_RESNET152,
#         config.MODEL_INCEPTION_V3,
#         config.MODEL_VGG16])
#     parser.add_argument('--port', type=int, default=config.server_address[1])
#     parser.add_argument('--host', type=str, default=config.server_address[0])
#     parser.add_argument('--debug', action='store_true')
#     args = parser.parse_args()
#
#     print("* Loading model and starting server. Please wait")
#     init(args.model)
#     model_module, model = load_model()
#     prediction_interpreter = PredictionInterpreter()
#     app.run(
#         host=args.host,
#         port=args.port,
#         debug=args.debug,
#         use_reloader=False
#     )
