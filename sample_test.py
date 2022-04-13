from PatchEnsemble import PatchEnsemble
import os

base_path = os.getcwd()
dataset = 'PCam'
classifier_type = 'patch'
degradation = ''
degradation_list = ['', '_degraded_0_13_to_0_10']

PatchEnsemble = PatchEnsemble(dataset=dataset, classifier_type=classifier_type, degradation=degradation,
                              base_path=base_path)

for d in degradation_list:
    PatchEnsemble.test(batch_size=32, test_degradation=d)
