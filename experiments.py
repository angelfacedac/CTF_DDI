import numpy as np

from data import GetData
from method import Model

class Experiments(object):

    def __init__(self, drug_drug_data, model_name='CTF', **kwargs):
        super().__init__()
        self.drug_drug_data = drug_drug_data
        self.model = Model(model_name)
        self.parameters = kwargs


    def CV_triplet(self):
        k_folds = 5
        index_matrix = np.array(np.where(self.drug_drug_data.X == 1))
        positive_num = index_matrix.shape[1]
        sample_num_per_fold = int(positive_num / k_folds)

        np.random.seed(0)
        np.random.shuffle(index_matrix.T)

        metrics_tensor = np.zeros((1, 7))
        # metrics_CP = np.zeros((1, 7))
        for k in range(k_folds):

            train_tensor = np.array(self.drug_drug_data.X, copy=True)
            if k != k_folds - 1:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold:])


            train_tensor[train_index] = 0
            S1 = np.mat(self.drug_drug_data.S1)
            S2 = np.mat(self.drug_drug_data.S2)
            predict_tensor = self.model()(train_tensor, S1, S2,
                                          r=self.parameters['r'],
                                          mu=self.parameters['mu'], eta=self.parameters['eta'],
                                          alpha=self.parameters['alpha'], beta=self.parameters['beta'],
                                          lam=self.parameters['lam'],
                                          xita=self.parameters['xita'],
                                          tol=self.parameters['tol'], max_iter=self.parameters['max_iter'],
                                          # epoch=self.parameters['epoch'], batch_size=self.parameters['batch_size'],
                                          # hids_size=self.parameters['hids_size'],
                                          # lr=self.parameters['lr'], weight_decay=self.parameters['weight_decay'],
                                          # k=k, label=np.array(self.drug_drug_data.X, copy=True), train_index=train_index,
                                          # Y=np.array(self.drug_drug_data.X, copy=True)
                                          )


            for i in range(10):
                metrics_tensor = metrics_tensor + self.cv_tensor_model_evaluate(self.drug_drug_data.X, predict_tensor, train_index, i)

        # print(metrics_tensor / (k + 1))
        result = np.around(metrics_tensor / 50, decimals=4)
        return result

    def cv_tensor_model_evaluate(self, association_tensor, predict_tensor, train_index, seed):
        test_po_num = np.array(train_index).shape[1]
        test_index = np.array(np.where(association_tensor == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index.T)
        # print(np.where((negative_index-test_index)!=0))
        test_ne_index = tuple(test_index[:, :test_po_num])
        real_score = np.column_stack(
            (np.mat(association_tensor[test_ne_index].flatten()), np.mat(association_tensor[train_index].flatten())))
        predict_score = np.column_stack(
            (np.mat(predict_tensor[test_ne_index].flatten()), np.mat(predict_tensor[train_index].flatten())))
        # real_score and predict_score are array
        return self.get_metrics(real_score, predict_score)

    def get_metrics(self, real_score, predict_score):
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]
        return aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision








if __name__ == '__main__':

    drug_drug_data = GetData()

    experiment = Experiments(drug_drug_data, model_name='CTF', r=51, mu=0.5, eta=0.2, alpha=0.5, beta=0.5, lam=0.5, xita=0.5, tol=1e-6, max_iter=200)

    # experiment = Experiments(drug_drug_data, model_name='TSNE_CTF', r=51, mu=0.5, eta=0.2, alpha=0.5, beta=0.5, lam=0.5, xita=0.5, tol=1e-6, max_iter=200)

    # experiment = Experiments(drug_drug_data, model_name='CTF_DNN', r=51, mu=0.5, eta=0.2, alpha=0.5, beta=0.5, lam=0.5, xita=0.5, tol=1e-6, max_iter=200,
    #                          epoch=300, batch_size=1000, hids_size=[256, 256, 128], lr=0.0001, weight_decay=5e-4)


    # experiment = Experiments(drug_drug_data, model_name='TFAI_CP_within_mod', r=4, alpha=0.5, beta=2.0, lam=0.001, tol=1e-6, max_iter=200)

    # experiment = Experiments(drug_drug_data, model_name='TDRC', r=4, alpha=0.125, beta=1.0, lam=0.001, tol=1e-6, max_iter=200)

    # experiment = Experiments(drug_drug_data, model_name='CP', r=4, alpha=None, beta=None, lam=None, tol=1e-6, max_iter=200)

    # for r in range(1, 100):
    # # for mu in [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 0.75]:
    # #     for eta in [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 0.75]:
    # #         for alpha in [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 0.75]:
    # #             for beta in [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 0.75]:
    # #                 for lam in [0.001, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 0.75]:
    #     experiment = Experiments(drug_drug_data, model_name='CTF',
    #                              r=r, mu=0.1, eta=0.1, alpha=0.1, beta=0.175, lam=0.5, xita=0.01,
    #                              tol=1e-6, max_iter=200
    #                              )
    #     print(f"mu={0.1}\teta={0.1}\talpha={0.1}\tbata={0.175}\tlam={0.5}\txita={0.01}")
    #     print('\t'.join(map(str, experiment.CV_triplet()[0])))

    # for mu in [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 0.75]:
    #     experiment = Experiments(drug_drug_data, model_name='CTF',
    #                              r=r, mu=0.1, eta=0.1, alpha=0.1, beta=0.175, lam=0.5, xita=0.01,
    #                              tol=1e-6, max_iter=200
    #                              )
    #     print(f"mu={mu}\teta={0.1}\talpha={0.1}\tbata={0.175}\tlam={0.5}\txita={0.01}")
    #     print('\t'.join(map(str, experiment.CV_triplet()[0])))
    #
    # for eta in [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 0.75]:
    #     experiment = Experiments(drug_drug_data, model_name='CTF',
    #                              r=r, mu=0.1, eta=0.1, alpha=0.1, beta=0.175, lam=0.5, xita=0.01,
    #                              tol=1e-6, max_iter=200
    #                              )
    #     print(f"mu={0.1}\teta={eta}\talpha={0.1}\tbata={0.175}\tlam={0.5}\txita={0.01}")
    #     print('\t'.join(map(str, experiment.CV_triplet()[0])))
    #
    # for alpha in [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 0.75]:
    #     experiment = Experiments(drug_drug_data, model_name='CTF',
    #                              r=51, mu=0.1, eta=0.1, alpha=0.1, beta=0.175, lam=0.5, xita=0.01,
    #                              tol=1e-6, max_iter=200
    #                              )
    #     print(f"mu={0.1}\teta={0.1}\talpha={alpha}\tbata={0.175}\tlam={0.5}\txita={0.01}")
    #     print('\t'.join(map(str, experiment.CV_triplet()[0])))

    # experiment = Experiments(drug_drug_data, model_name='CTF', r=51, mu=0.5, eta=0.2, alpha=0.5, beta=0.5, lam=0.5,
    #                          xita=0.5, tol=1e-6, max_iter=200)

    # for r in range(1, 100):
    #     experiment = Experiments(drug_drug_data, model_name='CP', r=r, alpha=None, beta=None, lam=None, tol=1e-6,
    #                              max_iter=200)
    #     print(f"r={r}")
    #     print('\t'.join(map(str, experiment.CV_triplet()[0])))

    print('\t'.join(map(str, experiment.CV_triplet()[0])))
