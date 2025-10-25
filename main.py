"""End-to-end entry point for the loan approval data mining workflow."""
from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from association_rules import mine_association_rules, prepare_transactions, plot_rule_network
from clustering_analysis import plot_clusters, run_kmeans
from data_preprocessing import PreprocessingArtifacts, preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "loan_approval.csv"


def run_classification_model(artifacts: PreprocessingArtifacts) -> str:
    """Train a baseline classifier to predict loan approval status."""
    target_col = None
    for candidate in ("Loan_Status", "loan_status", "Approved"):
        if candidate in artifacts.raw_data.columns:
            target_col = candidate
            break
    if target_col is None:
        raise ValueError("Unable to find the loan approval target column in the dataset.")

    X = artifacts.cleaned_data
    y = artifacts.raw_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report


def main() -> None:
    """Execute preprocessing, clustering, association rules, and classification."""
    artifacts = preprocess_data(str(DATA_PATH))

    print("=== Exploratory Data Analysis ===")
    from data_preprocessing import initial_eda  # Imported lazily to avoid circular

    eda_results = initial_eda(artifacts.raw_data)
    for name, df in eda_results.items():
        print(f"--- {name.upper()} ---")
        print(df.head())

    print("\n=== Clustering Analysis ===")
    clustering_result = run_kmeans(artifacts.cleaned_data)
    print("Clustering metrics:", clustering_result.metrics)
    plot_clusters(clustering_result, output_path=str(PROJECT_ROOT / "cluster_plot.png"))
    print("Cluster plot saved to cluster_plot.png")

    print("\n=== Association Rule Mining ===")
    transaction_df = prepare_transactions(artifacts.cleaned_data)
    association_result = mine_association_rules(transaction_df)
    print("Top association rules:")
    print(association_result.rules.head())
    plot_rule_network(
        association_result.rules,
        output_path=str(PROJECT_ROOT / "association_rules_network.png"),
    )
    print("Association rule network saved to association_rules_network.png")

    print("\n=== Classification Model ===")
    report = run_classification_model(artifacts)
    print(report)


if __name__ == "__main__":
    main()
