
from mkl.data_preprocessing import normalization, scale_data
from sklearn.model_selection import train_test_split
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from mkl.data_preprocessing import load_X, load_Y, load_and_process
from mkl.evaluate import kernel_similarity
import numpy as np
from tabulate import tabulate
from mkl.model import Model
from mkl.util import process


def linear_kernel(X, Z=None, normalize=False):
    return X @ Z.T


def identity_kernel(n):
    return torch.diag(torch.ones(n,  dtype=torch.double))


def compute_kernels(Xtr, Xte, Ytr):

    KLtr = [linear_kernel(Xtr, Xtr, None)**d for d in range(1, 13)] + [identity_kernel(len(Ytr))]
    KLte = [linear_kernel(Xte, Xtr,None)**d for d in range(1, 13)]
    KLte.append(torch.zeros((KLte[0]).size()))

    return KLtr, KLte, Ytr


def structured_sparsity(KLtr, KLte, Yte, Ytr):
    print("REGULARIZED")

    base_learner = SVC(C=0.5)
    # clf = MEMO(base_learner)
    clf = Model()
    clf = clf.fit(KLtr, Ytr)

    y_pred = clf.predict(KLtr)
    train_accuracy = accuracy_score(Ytr, y_pred)

    y_pred = clf.predict(KLte)
    test_accuracy = accuracy_score(Yte, y_pred)

    return train_accuracy, test_accuracy



def load_data():
    #processed and saved data uploaded
    # X, Y = load_and_process()

    # Load
    X = np.load('X.npy')
    Y = np.load('Y.npy')


    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.5, random_state=42)

    return Xtr, Xte, Ytr, Yte

def svm(Xtr, Xte, Ytr, Yte):
    print("SVM")
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(Xtr, Ytr)
    y_pred_train = svm_classifier.predict(Xtr)
    y_pred_test = svm_classifier.predict(Xte)
    train_accuracy = accuracy_score(Ytr, y_pred_train)
    test_accuracy = accuracy_score(Yte, y_pred_test)
    return train_accuracy, test_accuracy


def svm_with_kernel(KLtr, KLte, Ytr, Yte):
    print("SVM WITH KERNEL")
    svm_classifier = SVC()
    svm_classifier.fit(KLtr, Ytr)
    y_pred_train = svm_classifier.predict(KLtr)
    y_pred_test = svm_classifier.predict(KLte)
    train_accuracy = accuracy_score(Ytr, y_pred_train)
    test_accuracy = accuracy_score(Yte, y_pred_test)
    return train_accuracy, test_accuracy

def kernel_similarity_table(KLtr):
    num_kernels = len(KLtr)
    similarity_table = np.zeros((num_kernels, num_kernels))

    for i in range(num_kernels):
        for j in range(num_kernels):
            similarity_table[i, j] = kernel_similarity(KLtr[i], KLtr[j])

    return similarity_table


def main():
    Xtr, Xte, Ytr, Yte = load_data()
    svm(Xtr, Xte, Ytr, Yte)
    Xtr = process(Xtr)
    Xte = process(Xte)
    Ytr = process(Ytr)
    Yte = process(Yte)

    KLtr, KLte, Ytr = compute_kernels(Xtr, Xte, Ytr)
    print("frob : ", kernel_similarity(KLtr[0], KLtr[1]))

    results = []

    # SVM
    train_accuracy_svm, test_accuracy_svm = svm(Xtr, Xte, Ytr, Yte)
    results.append(("SVM", train_accuracy_svm, test_accuracy_svm))

    # SVM with Kernel
    train_accuracy, test_accuracy = svm_with_kernel(KLtr[0], KLte[0], Ytr, Yte)
    results.append(("SVM with Kernel", train_accuracy, test_accuracy))

    # Structured Sparsity
    train_accuracy, test_accuracy = structured_sparsity(KLtr, KLte, Ytr, Yte)
    results.append(("MKL", train_accuracy, test_accuracy))


    # similarity_table = kernel_similarity_table(KLtr)
    #
    # table_headers = [''] + [f'Kernel {i + 1}' for i in range(len(KLtr))]
    #
    # table_rows = [[f'Kernel {i + 1}'] + list(similarity_table[i]) for i in range(len(KLtr))]
    #
    # similarity_table_formatted = tabulate(table_rows, headers=table_headers, tablefmt="grid")
    #
    # print("Similarity Table:")
    # print(similarity_table_formatted)

    print("\nResults Table:")
    print(tabulate(results, headers=["Model", "Train Accuracy", "Test Accuracy"]), "\n")



if __name__ == "__main__":
    main()
