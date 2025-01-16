_document_types = [
    "general",
    "academic",
]

_entity_labels = {
    "general": [
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    ],
    "academic": [
        # Core paper components
        "Paper",
        "Abstract",
        "Method",
        "Result",
        "Conclusion",
        # Research elements
        "Research_Question",
        "Hypothesis",
        "Experiment",
        "Dataset",
        "Algorithm",
        "Model",
        # Metadata
        "Author",
        "Institution",
        "Journal",
        "Conference",
        "Year",
        "Keywords",
        # Domain concepts
        "Scientific_Concept",
        "Technology",
        "Metric",
    ],
}

_relation_labels = {
    "general": [
        "ORG-AFF",
        "PART-WHOLE",
        "PER-SOC",
        "GEN-AFF",
        "PHYS",
        "ART",
        "PER-SOC",
        "PER-ORG",
        "ORG-ORG",
        "ORG-LOC",
        "LOC-LOC",
        "LOC-ORG",
        "LOC-PER",
        "PER-LOC",
    ],
    "academic": [
        # Structural relations
        "CONTAINS",  # Paper CONTAINS Method
        "PART_OF",  # Method PART_OF Paper
        # Research relations
        "PROPOSES",  # Paper PROPOSES Method
        "EVALUATES",  # Paper EVALUATES Dataset
        "PROVES",  # Experiment PROVES Hypothesis
        "ACHIEVES",  # Method ACHIEVES Result
        # Citation relations
        "CITES",  # Paper CITES Paper
        "BUILDS_ON",  # Paper BUILDS_ON Paper
        "COMPARES_WITH",  # Paper COMPARES_WITH Paper
        # Metadata relations
        "AUTHORED_BY",  # Paper AUTHORED_BY Author
        "AFFILIATED_WITH",  # Author AFFILIATED_WITH Institution
        "PUBLISHED_IN",  # Paper PUBLISHED_IN Journal
        # Concept relations
        "USES",  # Method USES Technology
        "IMPROVES",  # Method IMPROVES Metric
        "RELATES_TO",  # Concept RELATES_TO Concept
    ],
}


def get_default_lables(document_type: str = "general") -> tuple[list[str], list[str]]:
    """
    Get the default entity and relation labels for a given document type

    Args:
        document_type (str, optional): The type of document. Defaults to "general".

    Returns:
        tuple[list[str], list[str]]: The entity and relation labels
    """
    return _entity_labels[document_type], _relation_labels[document_type]
