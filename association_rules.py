"""Association rule mining module using FP-Growth."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth


@dataclass
class AssociationRulesResult:
    """Container for frequent itemsets and derived rules."""

    frequent_itemsets: pd.DataFrame
    rules: pd.DataFrame


def prepare_transactions(data: pd.DataFrame, categorical_prefix: str = "cat_") -> pd.DataFrame:
    """Convert wide one-hot encoded data into transaction format for FP-Growth."""
    transaction_df = data.copy()
    # Ensure binary encoding (values > 0 become 1)
    transaction_df = (transaction_df > 0).astype(int)
    transaction_df.columns = [f"{categorical_prefix}{col}" for col in transaction_df.columns]
    return transaction_df


def mine_association_rules(
    data: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
) -> AssociationRulesResult:
    """Run FP-Growth and generate strong association rules."""
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    ).sort_values(by=["support", "confidence", "lift"], ascending=False)
    return AssociationRulesResult(frequent_itemsets=frequent_itemsets, rules=rules)


def plot_rule_network(rules: pd.DataFrame, top_n: int = 10, output_path: str | None = None) -> None:
    """Visualize association rules as a directed network graph."""
    top_rules = rules.head(top_n)
    graph = nx.DiGraph()

    for _, row in top_rules.iterrows():
        antecedent = ", ".join(sorted(list(row["antecedents"])))
        consequent = ", ".join(sorted(list(row["consequents"])))
        weight = row["lift"]
        graph.add_node(antecedent, type="antecedent")
        graph.add_node(consequent, type="consequent")
        graph.add_edge(antecedent, consequent, weight=weight, support=row["support"])

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color="#91c8f6")
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=20, edge_color="#2f4858")
    nx.draw_networkx_labels(graph, pos, font_size=10)
    edge_labels = {(
        u,
        v,
    ): f"lift={d['weight']:.2f}\nsupport={d['support']:.2f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Association Rule Network (Top Rules)")
    plt.axis("off")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def main() -> None:
    """Example usage running association rule mining."""
    from data_preprocessing import preprocess_data

    artifacts = preprocess_data("loan_approval.csv")
    transaction_df = prepare_transactions(artifacts.cleaned_data)
    result = mine_association_rules(transaction_df)
    print(result.rules.head())
    plot_rule_network(result.rules)


if __name__ == "__main__":
    main()
