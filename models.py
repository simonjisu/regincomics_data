# coding utf-8
import readdata
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, KFold
from io import StringIO
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz

class Run(object):
    def __init__(self, settings):
        self.settings = settings

    def initiate(self):
        self.regin = readdata.Regin()

        # training data initiating
        self.regin.read_data(self.settings['train_fileloc'])
        self.train_X, self.train_y, self.train_data = self.regin.return_data()
        self.train_privacy_comic = self.regin.privacy_comic
        self.train_purchasing_power = self.regin.purchasing_power

        # test data initiating
        self.regin.read_data(self.settings['test_fileloc'], train_switch=False)
        self.test_X, self.test_data = self.regin.return_data()
        self.test_privacy_comic = self.regin.privacy_comic
        self.test_purchasing_power = self.regin.purchasing_power

        # CV settings
        self.cv = KFold(self.settings['n_kfold'])
        self.preference_max_depth = 0
        self.classifier = None

    def forest(self):
        self.classifier = 'Random Forest'
        model_forest = ExtraTreesClassifier(criterion=self.settings['criterion'], n_estimators=self.settings['n_estimator'])
        model_forest.fit(self.train_X, self.train_y)

        std = np.std([tree.feature_importances_ for tree in model_forest.estimators_], axis=0)
        importances = model_forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        column_list = self.train_X.columns.tolist()
        gage = 0

        print('Feature importance')
        print('==' * 40)
        for f in range(self.train_X.shape[1]):
            if gage <= self.settings['accuracy_gage']:
                print('{0}. feature {1}: {2} ({3:.6f})'.format(f + 1, indices[f], column_list[indices[f]], importances[indices[f]]))
                gage += importances[indices[f]]
                self.preference_max_depth += 1

        plt.title("Feature importances")
        plt.bar(range(self.train_X.shape[1])[0:20], importances[indices][0:20], color="r", yerr=std[indices][0:20], align="center")
        plt.xticks(range(self.train_X.shape[1])[0:20], indices[0:20])
        plt.show()

        return model_forest

    def tree(self):
        self.classifier = 'Decision Tree'
        model_tree = DecisionTreeClassifier(criterion=self.settings['criterion'], max_depth=self.preference_max_depth)
        model_tree.fit(self.train_X, self.train_y)
        return model_tree

    def CV_test(self, classifier):
        yhat = classifier.predict(self.train_X)
        con_mat = confusion_matrix(self.train_y, yhat)
        fpr, tpr, thresholds = roc_curve(self.train_y, yhat)
        score = cross_val_score(classifier, self.train_X, self.train_y, scoring='accuracy', cv=self.cv)
        mean_score = score.mean()
        auc_score = roc_auc_score(self.train_y, yhat)
        print('ROC Auc Score for model-{0:} is {1:.4f}'.format(self.classifier, auc_score))
        print('Corss Validation Score for model-{0:} is {1:.4f}'.format(self.classifier, mean_score))

        if self.settings['report']:
            print('==' * 40)
            print('Training Result')
            print('==' * 40)
            print('Confusion Matrix')
            print(con_mat)
            print('--' * 40)
            print('Classification Report')
            print(classification_report(self.train_y, yhat, target_names=['notbuy', 'buy']))
            print('--' * 40)
            print('Accuracy Score')
            print(accuracy_score(self.train_y, yhat))
            print('--' * 40)
            plt.plot(fpr, tpr)
            plt.xlabel('Fall Out Rate')
            plt.ylabel('Recall Rate')
            plt.title('ROC Curve for model{}(Training)'.format(self.classifier))
            plt.show()

    def test(self, classifier):
        return classifier.predict(self.test_X)

    def draw_decision_tree(self, classifier):
        dot_buf = StringIO()
        export_graphviz(classifier, out_file=dot_buf, feature_names=self.train_X.columns)
        graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
        image = graph.create_png()
        return Image(image)

    def export_tree_dot(self, classifier):
        export_graphviz(classifier, out_file='tree.dot', feature_names=self.train_X.columns)

