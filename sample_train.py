from PatchEnsemble import PatchEnsemble
import os

base_path = os.getcwd()
dataset = 'PCam'
classifier_type = 'patch'
degradation = '_degraded_0_13_to_0_10'
train_top = False
extract_features = True
fine_tune = True
model_name_suffix = '-hyper-parameter-test'

# Hyper parameters
batch_size = 32
learning_rate = 1e-4
epochs = 100
num_fc_layers = 128
dropout_rate = 0.5
vgg_layer = 16
mnet_layer = 97
dnet_layer = 480
validation_split = 0.25

PatchEnsemble = PatchEnsemble(dataset=dataset, classifier_type=classifier_type, degradation=degradation,
                              base_path=base_path, num_fc_layers=num_fc_layers, dropout_rate=dropout_rate,
                              vgg_layer=vgg_layer, mnet_layer=mnet_layer, dnet_layer=dnet_layer,
                              validation_split=validation_split, model_name_suffix=model_name_suffix)

if train_top:
    PatchEnsemble.train_top(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs,
                            extract_features=extract_features)

if fine_tune:
    PatchEnsemble.fine_tune(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)
