import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def line(title=""):
    print("\n" + "=" * 60)
    if title:
        print(title)
        print("=" * 60)


# ----------------------------------------------------------------------
# STEP 1: LOAD DATASET
# ----------------------------------------------------------------------
line("STEP 1: LOAD DATASET")

iris = load_iris()

# Put the data into a DataFrame so it is easy to read
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in iris.target]

print(f"Dataset name  : Iris flower dataset")
print(f"Total samples : {df.shape[0]}")
print(f"Features      : {df.shape[1] - 1}")
print(f"Classes       : {', '.join(iris.target_names)}")

print("\nFirst 5 rows:")
print(df.head())


# ----------------------------------------------------------------------
# STEP 2: EXPLORE THE DATA
# ----------------------------------------------------------------------
line("STEP 2: EXPLORE THE DATA")

print("Samples per class:")
print(df["species"].value_counts())

print("\nBasic statistics:")
print(df.describe().round(2))

print("\nMissing values per column:")
print(df.isnull().sum())


# ----------------------------------------------------------------------
# STEP 3: SPLIT INTO TRAINING AND TESTING SETS
# ----------------------------------------------------------------------
line("STEP 3: SPLIT THE DATA")

X = df[iris.feature_names]   # features (input)  - kept as a DataFrame
y = iris.target              # labels   (output)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% kept aside for testing
    random_state=42,      # fixed seed so results are reproducible
    stratify=y            # keep class balance in both sets
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


# ----------------------------------------------------------------------
# STEP 4: TRAIN THE MODEL
# ----------------------------------------------------------------------
line("STEP 4: TRAIN THE MODEL")

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("Model type : Decision Tree Classifier")
print("Max depth  : 3")
print("Status     : trained successfully")


# ----------------------------------------------------------------------
# STEP 5: MAKE PREDICTIONS
# ----------------------------------------------------------------------
line("STEP 5: MAKE PREDICTIONS ON TEST DATA")

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual": [iris.target_names[i] for i in y_test],
    "Predicted": [iris.target_names[i] for i in y_pred],
})
results["Correct"] = results["Actual"] == results["Predicted"]

print(results.to_string(index=False))


# ----------------------------------------------------------------------
# STEP 6: EVALUATE THE MODEL
# ----------------------------------------------------------------------
line("STEP 6: EVALUATE THE MODEL")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}  ({accuracy * 100:.1f}%)")

print("\nConfusion matrix (rows = actual, columns = predicted):")
cm = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=iris.target_names,
    columns=iris.target_names,
)
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Feature importance (how much each feature influenced the decisions):")
for name, score in zip(iris.feature_names, model.feature_importances_):
    print(f"  {name:<22} {score:.3f}")

print("\nRules the model learned:")
print(export_text(model, feature_names=list(iris.feature_names)))


# ----------------------------------------------------------------------
# STEP 7: PREDICT A NEW, UNSEEN FLOWER
# ----------------------------------------------------------------------
line("STEP 7: PREDICT A NEW SAMPLE")

# [sepal length, sepal width, petal length, petal width] in cm
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names,
)

predicted_class = model.predict(new_flower)[0]
probabilities = model.predict_proba(new_flower)[0]

print("New measurements:")
for name, value in zip(iris.feature_names, new_flower.iloc[0]):
    print(f"  {name:<22} {value} cm")

print(f"\nPredicted species : {iris.target_names[predicted_class]}")
print("Confidence per class:")
for name, prob in zip(iris.target_names, probabilities):
    print(f"  {name:<12} {prob * 100:.1f}%")

line("PROGRAM FINISHED")
 