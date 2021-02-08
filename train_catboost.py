import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
import matplotlib
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


class Model(object):
    def __init__(self):
        self.data = pd.read_csv('./data/features2.csv')
        self.df1 = pd.DataFrame(self.data)  # use tx only
        self.df2 = pd.DataFrame(self.data)  # use opcode only
        self.df3 = pd.DataFrame(self.data)  # use tx + opcode
        self.df4 = pd.DataFrame(self.data)

        self.features1 = self.df1.drop(
            ['address', 'GASLIMIT', 'EXP', 'CALLDATALOAD', 'SLOAD', 'CALLER', 'LT', 'GAS', 'MOD', 'MSTORE', 'ponzi'], 1)
        self.features2 = self.df2.drop(
            ['address', 'kr', 'bal', 'n_inv', 'n_pay', 'pr', 'n_max', 'd_ind', 'tra_inv', 'tra_pay', 'n_invor',
             'n_payer', 'std_pay', 'ponzi'], 1)
        self.features3 = self.df3.drop(['address', 'ponzi'], 1)
        self.labels = self.df4.drop(
            ['address', 'kr', 'bal', 'n_inv', 'n_pay', 'pr', 'n_max', 'd_ind', 'tra_inv', 'tra_pay', 'n_invor',
             'n_payer', 'std_pay', 'GASLIMIT', 'EXP', 'CALLDATALOAD', 'SLOAD', 'CALLER', 'LT', 'GAS', 'MOD', 'MSTORE'],
            1)

        self.scoring = {
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1_score': make_scorer(f1_score)
        }

        self.build_models()

    def build_models(self):
        print("\n######################## Training the model using only transaction features ########################")
        x_train, x_test, y_train, y_test = train_test_split(self.features1.iloc[:, :-1], self.labels.iloc[:, -1],
                                                            test_size=0.3, random_state=1234)

        model1 = CatBoostClassifier()
        # fit the model with the training data
        model1.fit(x_train, y_train)

        # 5-fold cross validation

        kfold = KFold(n_splits=10, shuffle=True, random_state=123)
        results1 = cross_validate(model1, self.features1, self.labels, cv=kfold, scoring=self.scoring)
        # print(results)

        print("\n######################## Training the model using only opcode features ########################")
        x_train, x_test, y_train, y_test = train_test_split(self.features2.iloc[:, :-1], self.labels.iloc[:, -1],
                                                            test_size=0.3, random_state=1234)

        model2 = CatBoostClassifier()
        # fit the model with the training data
        model2.fit(x_train, y_train)

        # 5-fold cross validation

        kfold = KFold(n_splits=10, shuffle=True, random_state=123)
        results2 = cross_validate(model2, self.features2, self.labels, cv=kfold, scoring=self.scoring)
        # print(results)

        print("\n######################## Training the model using combined features ########################")
        x_train, x_test, y_train, y_test = train_test_split(self.features3.iloc[:, :-1], self.labels.iloc[:, -1],
                                                            test_size=0.3, random_state=1234)

        model3 = CatBoostClassifier(one_hot_max_size=10, learning_rate=0.1, loss_function='Logloss',
                                    custom_loss=['Recall', 'Accuracy'], eval_metric='Accuracy')
        # fit the model with the training data
        model3.fit(x_train, y_train, plot=True)

        # 5-fold cross validation

        kfold = KFold(n_splits=11, shuffle=True, random_state=123)
        results3 = cross_validate(model3, self.features3, self.labels, cv=kfold, scoring=self.scoring)
        # print(results)

        print('precision: {:.2f}  recall: {:.2f}  f-score: {:.2f}'
              .format(results1['test_precision'].mean(),
                      results1['test_recall'].mean(), results1['test_f1_score'].mean()))

        print('precision: {:.2f}  recall: {:.2f}  f-score: {:.2f}'
              .format(results2['test_precision'].mean(),
                      results2['test_recall'].mean(), results2['test_f1_score'].mean()))

        print('precision: {:.2f}  recall: {:.2f}  f-score: {:.2f}'
              .format(results3['test_precision'].mean(),
                      results3['test_recall'].mean(), results3['test_f1_score'].mean()))

        # fea_ = model3.feature_importances_
        # fea_name = model3.feature_names_
        # plt.figure(figsize=(10, 10))
        # plt.barh(fea_name,fea_,height =0.5)

        # plt.savefig("tedian.jpg")

        # CatBoostClassifier.plot_metric(results3)
        # plt.show()

        # draw_tree('catboost_model3_file', 3)
        preds_raw = model3.predict(eval_data,prediction_type='RawFormulaVal')

        print(preds_raw)


if __name__ == '__main__':
    model_performance = Model()