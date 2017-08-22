# coding utf-8
import readdata
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
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

    def forest(self):
        model_forest = ExtraTreesClassifier(n_estimators=self.settings['n_estimator'])
        model_forest.fit(self.train_X, self.train_y)
        yhat = model_forest.predict(self.train_X)
        con_mat = confusion_matrix(self.train_y, yhat)
        fpr, tpr, thresholds = roc_curve(self.train_y, yhat)
        score = cross_val_score(model_forest, self.train_X, self.train_y, scoring='accuracy', cv=self.cv)
        mean_score = score.mean()
        print('Corss Validation Score for model-Random Forest is {0:.4f}'.format(mean_score))

        importances = model_forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        column_list = self.train_X.columns.tolist()
        gage = 0
        print('==' * 40)
        print('Feature importance')
        for f in range(self.train_X.shape[1]):
            if gage <= self.settings['accuracy_gage']:
                print('==' * 40)
                print('Feature importance')
                print('{0}. feature {1}: {2} ({3:.6f})'.format(f + 1, indices[f], column_list[indices[f]], importances[indices[f]]))
                gage += importances[indices[f]]
                self.preference_max_depth += 1

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
            plt.title('ROC Curve for model Random Forest(Training)')
            plt.show()

        return model_forest


    def tree(self):
        model_tree = DecisionTreeClassifier(criterion='gini', max_depth=self.preference_max_depth).fit(self.train_X, self.train_y)
        yhat = model_tree.predict(self.train_X)
        con_mat = confusion_matrix(self.train_y, yhat)
        fpr, tpr, thresholds = roc_curve(self.train_y, yhat)
        score = cross_val_score(model_tree, self.train_X, self.train_y, scoring='accuracy', cv=self.cv)
        mean_score = score.mean()
        print('Corss Validation Score for model-Decision Tree is {0:.4f}'.format(mean_score))

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
            plt.title('ROC Curve for model Random Forest(Training)')
            plt.show()

        return model_tree

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

