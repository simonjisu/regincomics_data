# coding utf-8
import models

settings = {'n_kfold': 10,
            'n_estimator': 10,
            'accuracy_gage': 0.8,
            'report': False,
            'train_fileloc': './data/lezhin_dataset_v2_training.tsv',
            'test_fileloc': './data/lezhin_dataset_v2_test_without_label.tsv'}

sess = models.Run(settings)
sess.initiate()
model_forest = sess.forest()
model_tree = sess.tree()
guess = sess.test(model_tree)
