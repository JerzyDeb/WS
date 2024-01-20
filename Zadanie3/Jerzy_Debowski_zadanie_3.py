from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
wine = datasets.load_wine()


def test_knn(data, target, k_values, distances):
    results = {}
    for k in k_values:
        for distance in distances:
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

            knn = KNeighborsClassifier(n_neighbors=k, metric=distance)
            knn.fit(X_train, y_train)

            predictions = knn.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            results[(k, distance)] = accuracy, report

    return results


k_values = [1, 2, 3]
distances = ['euclidean', 'manhattan', 'chebyshev']

iris_results = test_knn(iris.data, iris.target, k_values, distances)
wine_results = test_knn(wine.data, wine.target, k_values, distances)


best_iris = max(iris_results, key=iris_results.get)
best_wine = max(wine_results, key=wine_results.get)

print(best_iris)
print(iris_results[best_iris])


print(best_wine)
print(wine_results[best_wine])
