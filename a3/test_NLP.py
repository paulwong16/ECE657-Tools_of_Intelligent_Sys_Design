from pickle import load
import utlis
import torch
import numpy as np
from sklearn import metrics


if __name__ == "__main__":
    test_corpus, test_labels = utlis.get_data('./data/aclImdb/test/neg/*.txt', './data/aclImdb/test/pos/*.txt')
    test_corpus = utlis.normalize(test_corpus)

    saved_model = load(open('./models/20856733_NLP_model.pkl', 'rb'))

    model = saved_model['model']
    tfidf_vectorizer = saved_model['tf_vector']

    tfidf_test_features = tfidf_vectorizer.transform(test_corpus)
    X_test = tfidf_test_features.toarray()

    batch_size = 512
    batch_num_test = X_test.shape[0] // batch_size + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    correct = 0
    test_loss = 0
    pred_probas = []
    with torch.no_grad():
        for batch_idx in range(batch_num_test):
            start = batch_size * batch_idx
            end = min(X_test.shape[0], batch_size * (batch_idx + 1))
            data = torch.from_numpy(X_test[start:end]).float().to(device)
            target = torch.from_numpy(np.array(test_labels[start:end])).long().to(device)
            output = model(data)
            proba = torch.sigmoid(output)
            pred_probas += proba.cpu().detach().numpy().reshape(-1, 2).tolist()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    pred_probas = np.array(pred_probas)
    accuracy = correct / X_test.shape[0]
    print('Test Accuracy: %f' % accuracy)
    print('F1 score: %f' % metrics.f1_score(test_labels, np.argmax(pred_probas, axis=1)))
    print('Recall rate: %f' % metrics.recall_score(test_labels, np.argmax(pred_probas, axis=1)))

    fpr, tpr, threshold = metrics.roc_curve(test_labels, pred_probas[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    import matplotlib.pyplot as plt

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    precision, recall, _ = metrics.precision_recall_curve(test_labels, pred_probas[:, 1])
    plt.plot(recall, precision)
    plt.title('Precision-Recall')
    plt.xlim([0, 1])
    plt.ylim([0.5, 1])
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.show()
