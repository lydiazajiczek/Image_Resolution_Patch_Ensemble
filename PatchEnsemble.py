import os
import csv
import numpy as np
from params import get_param_dict
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, MobileNetV2, DenseNet201
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from sklearn.metrics import accuracy_score


def scheduler(epoch, lr):
    if epoch <= 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Learning rate scheduler which sets the learning rate according to epoch.
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = scheduler(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch {}: Learning rate is {}.".format(epoch, scheduled_lr))


# Custom callback at end of training epoch to decide when to save best model for whole_image classification
# Use of single random patches during training means validation accuracy is extremely variable
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, weights_path):
        super(SaveBestModel, self).__init__()
        self.best_weights = None
        self.best_acc = 0
        self.best_val_acc = 0
        self.best_both = 0
        self.weights_path = weights_path

    def on_epoch_end(self, epoch, logs):
        val_acc = logs['val_accuracy']
        acc = logs['accuracy']
        both = acc * (val_acc ** 2)
        if both > self.best_both:
            self.best_val_acc = val_acc
            self.best_acc = acc
            self.best_both = both
            print('\nBest accuracy/validation accuracy (epoch {}): {}%/{}%'.format(epoch, 100 * self.best_acc,
                                                                                   100 * self.best_val_acc))
            self.best_weights = self.model.get_weights()
            self.model.save_weights(self.weights_path)


# Divide input images into square patches of size (patch_size, patch_size) and return as a batch
# Used for testing patch-based whole-image classifier
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Note that batch size must be set to 1 for use in testing
        patches = tf.squeeze(tf.reshape(patches, [1, -1, self.patch_size, self.patch_size, 3]))
        return patches


# Return randomly selected square patch of size (patch_size, patch_size)
# Used for training patch-based whole-image classifier
class RandomPatch(tf.keras.layers.Layer):
    def __init__(self, rng, patch_size):
        super(RandomPatch, self).__init__()
        self.rng = rng
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_size, self.patch_size, 3])
        num_patches = int(images.shape[1] / self.patch_size) * int(images.shape[2] / self.patch_size)
        i = self.rng.integers(0, num_patches)
        random_patch = tf.squeeze(patches[:, i, :, :, :])
        return random_patch


# General PatchEnsemble class for patch-based image classification (single patches or whole-image) using
# ensemble of 3 CNNs
class PatchEnsemble:
    def __init__(self, dataset, degradation, classifier_type, base_path, num_fc_layers=128, dropout_rate=0.5,
                 vgg_layer=16, mnet_layer=97, dnet_layer=480, beta_1=0.6, beta_2=0.8, validation_split=0.25,
                 model_name_suffix=''):

        # Validate inputs and check paths
        self.dataset = dataset
        self.degradation = degradation
        self.base_path = base_path
        self.dataset_path = os.path.join(self.base_path, self.dataset, 'data' + self.degradation)
        if not (os.path.exists(self.dataset_path)):
            raise ValueError('Dataset path is not a valid path')
        self.train_path = os.path.join(self.dataset_path, "train")
        if not (os.path.exists(self.train_path)):
            raise ValueError('Training data path is not a valid path')
        self.val_path = os.path.join(self.dataset_path, "val")
        self.val_exists = os.path.exists(self.val_path)
        self.validation_split = validation_split
        self.test_path = os.path.join(self.dataset_path, "test")
        if not (os.path.exists(self.test_path)):
            raise ValueError('Test data path is not a valid path')

        # Hyper parameters
        self.random_seed = 1239
        self.num_fc_layers = num_fc_layers
        self.dropout_rate = dropout_rate
        self.vgg_layer = vgg_layer
        self.mnet_layer = mnet_layer
        self.dnet_layer = dnet_layer
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # Callbacks
        self.early_stopper = EarlyStopping(monitor='val_accuracy',
                                           min_delta=1e-4,
                                           patience=20)
        self.lr_reducer = ReduceLROnPlateau(monitor='loss',
                                            factor=np.sqrt(0.1),
                                            cooldown=0,
                                            patience=5,
                                            min_lr=0.5e-6)
        self.lr_scheduler = CustomLearningRateScheduler(scheduler)

        # Set dataset-specific and classifier type specific parameters (image size, model name etc)
        self.classifier_type = classifier_type
        params = get_param_dict(self.dataset)

        # Patch based classification only applies to PCam dataset
        if self.classifier_type == 'patch':
            if dataset != 'PCam':
                raise ValueError('Patch classifier type only compatible with PCam dataset')
            self.image_height = params['height']
            self.image_width = params['width']
            self.patch_input_shape = (self.image_height, self.image_width, 3)
            self.model_name = 'PatchEnsemble-' + self.dataset + self.degradation

        # All other datasets use whole-image classification
        elif self.classifier_type == 'whole_image':
            if dataset == 'PCam':
                raise ValueError('Whole image classifier type not compatible with PCam dataset')
            # Need to scale whole images to match d_eff of PCam and divide into integer number of PCam-sized patches
            PCam_params = get_param_dict('PCam')
            d_eff = params['d_pixel'] / (params['M_obj'] * params['M_relay'])
            d_eff_PCam = PCam_params['d_pixel'] / (PCam_params['M_obj'] * PCam_params['M_relay'])
            scale_factor = d_eff_PCam / d_eff
            self.image_height = int(
                ((params['height'] / scale_factor) // PCam_params['height']) * PCam_params['height'])
            self.image_width = int(
                ((params['width'] / scale_factor) // PCam_params['width']) * PCam_params['width'])
            self.input_shape = (self.image_height, self.image_width, 3)
            self.patch_input_shape = (PCam_params['height'], PCam_params['width'], 3)
            self.model_name = 'RandomPatchVoting-' + self.dataset + self.degradation
        else:
            raise ValueError('Unknown classifier type')

        # Create file paths for saving weights and bottleneck features
        self.weights_file = self.model_name + model_name_suffix
        self.weights_file_top = self.weights_file + '_top'
        self.bn_features = self.model_name + '-bottleneck_features'
        self.bn_feat_files = {'train_data_x': self.bn_features + '_train_data_x.npy',
                              'train_data_y': self.bn_features + '_train_data_y.npy',
                              'val_data_x': self.bn_features + '_val_data_x.npy',
                              'val_data_y': self.bn_features + '_val_data_y.npy'}

        # Create data generator classes
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        # In case of separate validation folder:
        if self.val_exists:
            self.train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                    rotation_range=90,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2,
                                                    horizontal_flip=True,
                                                    vertical_flip=True,
                                                    fill_mode='nearest',
                                                    )

            self.val_datagen = self.test_datagen

        # Otherwise use validation_split parameter:
        else:
            self.train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                    rotation_range=90,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2,
                                                    horizontal_flip=True,
                                                    vertical_flip=True,
                                                    fill_mode='nearest',
                                                    validation_split=self.validation_split,
                                                    )

            self.val_datagen = ImageDataGenerator(rescale=1. / 255,
                                                  validation_split=self.validation_split,
                                                  )

    # Different train and validation generators to return based on if validation folder exists or not
    def get_train_generator(self, batch_size, shuffle=True, augment=True):
        if augment:
            datagen = self.train_datagen
        else:
            datagen = self.val_datagen

        if self.val_exists:
            train_generator = datagen.flow_from_directory(self.train_path,
                                                          target_size=(
                                                              self.image_height, self.image_width),
                                                          batch_size=batch_size,
                                                          shuffle=shuffle,
                                                          class_mode='binary',
                                                          )
        else:
            train_generator = datagen.flow_from_directory(self.train_path,
                                                          target_size=(
                                                              self.image_height, self.image_width),
                                                          batch_size=batch_size,
                                                          shuffle=shuffle,
                                                          class_mode='binary',
                                                          subset='training',
                                                          seed=self.random_seed,
                                                          )

        return train_generator

    def get_val_generator(self, batch_size):
        if self.val_exists:
            val_generator = self.val_datagen.flow_from_directory(self.val_path,
                                                                 target_size=(
                                                                     self.image_height, self.image_width),
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 class_mode='binary',
                                                                 )
        else:
            val_generator = self.val_datagen.flow_from_directory(self.train_path,
                                                                 target_size=(
                                                                     self.image_height, self.image_width),
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 class_mode='binary',
                                                                 subset='validation',
                                                                 seed=self.random_seed,
                                                                 )

        return val_generator

    # Will sometimes test on images from different dataset i.e. baseline model on degraded test sets
    # Note that BACH does not have test labels so need to set class_mode
    def get_test_generator(self, batch_size, test_path, test_labels):
        if test_labels:
            test_generator = self.test_datagen.flow_from_directory(test_path,
                                                                   target_size=(
                                                                       self.image_height, self.image_width),
                                                                   batch_size=batch_size,
                                                                   shuffle=False,
                                                                   class_mode='binary',
                                                                   )
        else:
            test_generator = self.test_datagen.flow_from_directory(test_path,
                                                                   target_size=(
                                                                       self.image_height, self.image_width),
                                                                   batch_size=batch_size,
                                                                   shuffle=False,
                                                                   class_mode=None,
                                                                   )
        return test_generator

    # Main patch classifier architecture, ensemble of 3 CNNs concatenated together
    def build_ensemble(self, fine_tune=True):
        input_tensor = Input(shape=self.patch_input_shape)

        # Load all 3 models pretrained on imagenet
        vgg = VGG19(weights='imagenet',
                    include_top=False,
                    input_shape=self.patch_input_shape,
                    input_tensor=input_tensor,
                    )
        mnet = MobileNetV2(weights='imagenet',
                           include_top=False,
                           input_shape=self.patch_input_shape,
                           input_tensor=input_tensor,
                           )
        dnet = DenseNet201(weights='imagenet',
                           include_top=False,
                           input_shape=self.patch_input_shape,
                           input_tensor=input_tensor,
                           )

        # Unfreeze final blocks in each CNN for fine tuning (these layers can be tuned as hyper parameters)
        if fine_tune:
            for layer in vgg.layers[:self.vgg_layer]:
                layer.trainable = False
            for layer in mnet.layers[:self.mnet_layer]:
                layer.trainable = False
            for layer in dnet.layers[:self.dnet_layer]:
                layer.trainable = False
        # Otherwise freeze everything (for training top layer)
        else:
            for layer in vgg.layers:
                layer.trainable = False
            for layer in mnet.layers:
                layer.trainable = False
            for layer in dnet.layers:
                layer.trainable = False

        # Concatenate output of 3 CNNs
        x1 = vgg.output
        x1 = GlobalAveragePooling2D()(x1)
        x2 = mnet.output
        x2 = GlobalAveragePooling2D()(x2)
        x3 = dnet.output
        x3 = GlobalAveragePooling2D()(x3)
        merge = Concatenate()([x1, x2, x3])
        ensemble_model = models.Model(inputs=input_tensor, outputs=merge)

        return ensemble_model

    # Top classifier model (first trained separately on bottleneck features)
    def build_top_model(self):
        top_model = models.Sequential()
        top_model.add(Dense(self.num_fc_layers, activation='relu'))
        top_model.add(Dropout(self.dropout_rate))
        top_model.add(BatchNormalization())
        top_model.add(Dense(1, activation='sigmoid'))

        return top_model

    # Patch classifier model (ensemble + top classifier)
    def build_patch_model(self, fine_tune=True, training=True, use_baseline_weights=True):
        ensemble_model = self.build_ensemble(fine_tune=fine_tune)
        top_model = self.build_top_model()

        if training:
            if fine_tune:
                if use_baseline_weights:
                    top_weights = os.path.join(self.base_path, 'PCam', 'data', 'PatchEnsemble-PCam_top')
                else:
                    top_weights = os.path.join(self.dataset_path, self.weights_file_top)
                top_model.load_weights(top_weights)

        patch_model = models.Sequential()
        patch_model.add(ensemble_model)
        patch_model.add(top_model)
        if not training:
            patch_model.load_weights(os.path.join(self.dataset_path, self.weights_file)).expect_partial()

        return patch_model

    # Whole-image model which uses patch classifier model + average voting
    def build_whole_image_model(self, patch_weights='', training=True, fine_tune=True):
        input_tensor = Input(shape=self.input_shape)
        patch_model = models.Sequential()
        patch_model.add(self.build_ensemble(fine_tune=fine_tune))
        patch_model.add(self.build_top_model())
        patch_weights_filepath = os.path.join(self.base_path, 'PCam', 'data' + patch_weights,
                                              'PatchEnsemble-PCam' + patch_weights)
        if os.path.exists(patch_weights_filepath + '.index'):
            patch_model.load_weights(patch_weights_filepath)
        else:
            raise ValueError('Patch weights file does not exist.')

        # Use RandomPatch layer to select single patch from whole image for training
        if training:
            rng = np.random.default_rng(self.random_seed)
            patch = RandomPatch(rng, self.patch_input_shape[0])(input_tensor)
            pred = patch_model(patch)
            model = models.Model(inputs=input_tensor, outputs=pred)

        # Predict on all patches and use average voting for validation/testing
        else:
            patches = Patches(self.patch_input_shape[0])(input_tensor)
            preds = patch_model(patches)
            vote_model = models.Model(inputs=input_tensor, outputs=preds)
            vote_model.load_weights(os.path.join(self.dataset_path, self.weights_file)).expect_partial()
            votes_tensor = vote_model.output
            vote = Lambda(lambda x: tf.math.reduce_mean(x, axis=0))(votes_tensor)
            model = models.Model(inputs=input_tensor, outputs=vote)

        return model

    # Train top classifier layers on extracted bottleneck features (can load previously extracted features)
    # Note that this only applies for patch classifier as feature extraction is not compatible with
    # extracting random patches
    def train_top(self, epochs, batch_size, learning_rate, extract_features=True):
        if extract_features:
            bottleneck_model = self.build_ensemble(fine_tune=False)
            # Do not use data augmentation for feature extraction
            train_generator = self.get_train_generator(batch_size=batch_size, shuffle=False, augment=False)
            val_generator = self.get_val_generator(batch_size=batch_size)
            # Extract bottleneck features from training and validation data
            train_data_x = bottleneck_model.predict(train_generator,
                                                    batch_size=batch_size,
                                                    verbose=1,
                                                    )
            val_data_x = bottleneck_model.predict(val_generator,
                                                  batch_size=batch_size,
                                                  verbose=1,
                                                  )
            train_data_y = train_generator.labels
            val_data_y = val_generator.labels

            # Save for future use
            np.save(os.path.join(self.dataset_path, self.bn_feat_files['train_data_x']), train_data_x)
            np.save(os.path.join(self.dataset_path, self.bn_feat_files['val_data_x']), val_data_x)
            np.save(os.path.join(self.dataset_path, self.bn_feat_files['train_data_y']), train_data_y)
            np.save(os.path.join(self.dataset_path, self.bn_feat_files['val_data_y']), val_data_y)
        else:
            # Otherwise load previously extracted features
            train_data_x = np.load(os.path.join(self.dataset_path, self.bn_feat_files['train_data_x']))
            val_data_x = np.load(os.path.join(self.dataset_path, self.bn_feat_files['val_data_x']))
            train_data_y = np.load(os.path.join(self.dataset_path, self.bn_feat_files['train_data_y']))
            val_data_y = np.load(os.path.join(self.dataset_path, self.bn_feat_files['val_data_y']))

        # Train top classifier using extracted bottleneck features
        top_model = self.build_top_model()
        optimizer = optimizers.Adam(learning_rate=learning_rate,
                                    beta_1=self.beta_1,
                                    beta_2=self.beta_2)
        top_model.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
        model_checkpoint = ModelCheckpoint(os.path.join(self.dataset_path, self.weights_file_top),
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode='auto')
        callbacks = [self.early_stopper, self.lr_reducer, self.lr_scheduler, model_checkpoint,
                     CSVLogger(os.path.join(self.dataset_path, self.weights_file_top))]
        history = top_model.fit(train_data_x, train_data_y,
                                batch_size=batch_size,
                                verbose=1,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=(val_data_x, val_data_y))
        acc = np.asarray(history.history['accuracy'])
        val_acc = np.asarray(history.history['val_accuracy'])
        index = np.where(val_acc == np.max(val_acc))
        print('Best training/validation accuracy (epoch {}): {}%/{}%'.format(index[0],
                                                                             100 * acc[index[0]],
                                                                             100 * val_acc[index[0]]))

    # Fine tune either patch or whole-image classifier
    def fine_tune(self, epochs, batch_size, learning_rate, patch_weights='', use_baseline_weights=True):
        if self.classifier_type == 'patch':
            model = self.build_patch_model(fine_tune=True, training=True, use_baseline_weights=use_baseline_weights)
        else:
            model = self.build_whole_image_model(patch_weights=patch_weights, training=True,
                                                 fine_tune=True)

        optimizer = optimizers.Adam(learning_rate=learning_rate,
                                    beta_1=self.beta_1,
                                    beta_2=self.beta_2)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        if self.classifier_type == 'patch':
            model_checkpoint = ModelCheckpoint(os.path.join(self.dataset_path, self.weights_file),
                                               monitor='val_accuracy',
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='auto')
            callbacks = [self.early_stopper, self.lr_reducer, self.lr_scheduler, model_checkpoint,
                         CSVLogger(os.path.join(self.dataset_path, self.weights_file))]
        else:
            # Use a different model saving callback for whole_image classification
            # Since validation accuracy is so variable (particularly for BACH)
            callbacks = [self.early_stopper, self.lr_reducer, self.lr_scheduler,
                         SaveBestModel(os.path.join(self.dataset_path, self.weights_file)),
                         CSVLogger(os.path.join(self.dataset_path, self.weights_file))]

        train_generator = self.get_train_generator(batch_size=batch_size, shuffle=True, augment=True)
        val_generator = self.get_val_generator(batch_size=batch_size)
        history = model.fit(train_generator,
                            batch_size=batch_size,
                            verbose=1,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_generator)

        if self.classifier_type == 'patch':
            acc = np.asarray(history.history['accuracy'])
            val_acc = np.asarray(history.history['val_accuracy'])
            index = np.where(val_acc == np.max(val_acc))
            print(self.weights_file)
            print('Best training/validation accuracy (epoch {}): {}%/{}%'.format(index[0],
                                                                                 100 * acc[index[0]],
                                                                                 100 * val_acc[index[0]]))
            self.test(batch_size=batch_size)
        else:
            # Create test model using patch average voting
            test_model = self.build_whole_image_model(patch_weights=patch_weights, training=False)
            test_model.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

            # Compute actual training and validation accuracy
            train_generator = self.get_train_generator(batch_size=1, augment=False)
            val_generator = self.get_val_generator(batch_size=1)
            scores = test_model.evaluate(train_generator, verbose=1, batch_size=1)
            print(self.weights_file)
            print('Full training accuracy: {}%'.format(100 * scores[1]))
            scores = test_model.evaluate(val_generator, verbose=1, batch_size=1)
            print('Full validation accuracy: {}%'.format(100 * scores[1]))
            self.test(batch_size=1)

    # Test for a specific test set (e.g. degraded using baseline model) otherwise use test set belonging to dataset
    # model was trained on
    def test(self, batch_size, test_degradation='', threshold=0.5):
        if (threshold < 0) & (threshold > 1):
            raise ValueError('Threshold for binary classification must be between 0 and 1 inclusive')

        if self.classifier_type == 'whole_image':
            batch_size = 1
            model = self.build_whole_image_model(training=False)
        else:
            batch_size = batch_size
            model = self.build_patch_model(training=False)

        retrained = self.degradation != ''
        if retrained:
            pred_labels_name = self.model_name + '_retrained' + test_degradation
        else:
            pred_labels_name = self.model_name + '_baseline' + test_degradation

        # Set test path based on if model is predicting on same type of images it was trained on or if
        # baseline model is predicting on degraded test sets
        if test_degradation == '':
            # Test set in same folder as training set
            test_path = self.test_path
        else:
            # Baseline model testing on degraded test set
            test_path = os.path.join(self.base_path, self.dataset, "data" + test_degradation, "test")
            if not (os.path.isdir(test_path)):
                raise ValueError('Test path is not a valid path')

        test_generator = self.get_test_generator(batch_size=batch_size, test_path=test_path,
                                                 test_labels=('BreaKHis' in self.dataset))
        model.compile(loss=['binary_crossentropy'],
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])

        # Test procedure is different for each dataset
        if self.dataset == 'PCam':
            # Need to generate submission file for uploading to Kaggle to be scored
            y_pred = model.predict(test_generator, verbose=1)
            f_names = test_generator.filenames
            self.generate_submission_file(y_pred, f_names, os.path.join(self.dataset_path,
                                                                        pred_labels_name + '_submission.csv'))

        elif self.dataset == 'BACH':
            y_pred = np.squeeze(model.predict(test_generator, verbose=1))
            y_pred.tofile(os.path.join(self.dataset_path, pred_labels_name + '_test_pred.csv'), sep='\n')
            if retrained & (self.degradation != ''):
                # Load ground truth labels
                ground_truth_labels = os.path.join(self.base_path, self.dataset, 'data',
                                                   self.model_name + '_test_pred.csv')
                y_true = np.asarray(np.fromfile(ground_truth_labels, sep='\n') >= threshold).astype(float)
                y_pred = np.asarray(y_pred >= threshold).astype(float)
                y_pred.tofile(os.path.join(self.dataset_path, pred_labels_name))
                accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
                print(pred_labels_path + ' test accuracy: {}%'.format(100 * accuracy))

        # BreaKHis has all test labels
        else:
            scores = model.evaluate(test_generator, verbose=1)
            print(pred_labels_name)
            print('Test accuracy: {}%'.format(scores[1] * 100))

    # For uploading to Kaggle (PCam test scoring)
    @staticmethod
    def generate_submission_file(y_pred, f_names, submission_filename):
        with open(submission_filename, 'w', newline='') as c:
            writer = csv.writer(c, delimiter=',')
            writer.writerow(['id', 'label'])
            for i, fn in enumerate(f_names):
                fn = fn.split('/')[1]
                f_name, f_ext = os.path.splitext(fn)
                writer.writerow([f_name, float(y_pred[i])])
