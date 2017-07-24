from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import pandas as pd
import numpy as np
import config
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Base model architecture',
                        choices=[config.MODEL_RESNET50,
                                 config.MODEL_RESNET152,
                                 config.MODEL_INCEPTION_V3,
                                 config.MODEL_VGG16])
    parser.add_argument('--use_nn', action='store_true')
    args = parser.parse_args()
    return args


def encode(df):
    label_encoder = LabelEncoder().fit(df['class'])
    labels = label_encoder.transform(df['class'])
    classes = list(label_encoder.classes_)
    df = df.drop(['class'], axis=1)
    return df, labels, classes


def train_logistic():
    df = pd.read_csv(config.activations_path)
    df, y, classes = encode(df)

    X_train, X_test, y_train, y_test = train_test_split(df.values, y, test_size=0.2, random_state=17)

    params = {'C': [10, 2, .9, .4, .1], 'tol': [0.0001, 0.001, 0.0005]}
    log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
    clf = GridSearchCV(log_reg, params, scoring='neg_log_loss', refit=True, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("best params: " + str(clf.best_params_))
    print("Accuracy: ", accuracy_score(y_test, clf.predict(X_test)))

    setattr(clf, '__classes', classes)
    # save results for further using
    joblib.dump(clf, config.get_novelty_detection_model_path())


def train_nn():
    df = pd.read_csv(config.activations_path)
    df, y, classes = encode(df)
    X_train, X_test, y_train, y_test = train_test_split(df.values, y, test_size=0.2, random_state=17)

    model_module = util.get_model_class_instance()

    model = Sequential()
    model.add(Dense(48, input_dim=model_module.noveltyDetectionLayerSize, activation='elu', init='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='elu', init='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax', init='uniform'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    early_stopping = EarlyStopping(verbose=1, patience=40, monitor='val_loss')
    model_checkpoint = ModelCheckpoint(config.get_novelty_detection_model_path(), save_best_only=True,
                                       save_weights_only=True, monitor='val_loss')
    callbacks_list = [early_stopping, model_checkpoint]

    model.fit(
        X_train,
        y_train,
        nb_epoch=300,
        validation_data=(X_test, y_test),
        batch_size=16,
        callbacks=callbacks_list)

    out = model.predict(X_test)
    predictions = np.argmax(out, axis=1)
    print("Accuracy: ", accuracy_score(y_test, predictions))


if __name__ == '__main__':
    args = parse_args()
    config.model = args.model

    if not args.use_nn:
        train_logistic()
    else:
        train_nn()
