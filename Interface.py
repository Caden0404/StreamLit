import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Welcome to the Interface Module")

st.write("""
### Explore different classifier
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer", "Wine")
)

classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest")
)


def get_dataset(name):
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    return data.data, data.target, data.feature_names, data.target_names


X, y, feature_names, target_names = get_dataset(dataset_name)

st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(set(y)))


def add_parameter_ui(clf_name):
    params = {}

    if clf_name == "KNN":
        k = st.sidebar.slider("K", 1, 15, 5)
        leaf_size = st.sidebar.slider("Leaf Size", 1, 100, 30)

        params["K"] = k
        params["leaf_size"] = leaf_size

    elif clf_name == "SVM":
        c = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        degree = st.sidebar.slider("Degree", 1, 10, 3)
        kernel = st.sidebar.selectbox("Kernel", ("rbf", "linear", "poly"))

        params["C"] = c
        params["degree"] = degree
        params["kernel"] = kernel

    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15, 5)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100, 20)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["min_samples_split"] = min_samples_split

    return params


params = add_parameter_ui(classifier_name)
st.write("Classifier parameters:", params)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

if classifier_name == "KNN":
    model = KNeighborsClassifier(
        n_neighbors=params["K"],
        leaf_size=params["leaf_size"]
    )

elif classifier_name == "SVM":
    model = SVC(
        C=params["C"],
        degree=params["degree"],
        kernel=params["kernel"],
        random_state=42
    )

else:
    model = RandomForestClassifier(
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        min_samples_split=params["min_samples_split"],
        random_state=42
    )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("Model accuracy:", round(accuracy, 4))

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=target_names))

st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

st.write("PCA Plot:")

fig, ax = plt.subplots()
scatter = ax.scatter(x1, x2, c=y, cmap="viridis")

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)

st.pyplot(fig)
plt.close(fig)
