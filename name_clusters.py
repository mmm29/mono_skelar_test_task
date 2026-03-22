import re
from itertools import combinations
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


# ── helpers ──────────────────────────────────────────────────────────────────

TITLES = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "sir",
    "lady",
    "lord",
    "rev",
    "capt",
}


def normalize_name(name: str) -> str:
    """Lowercase, remove punctuation, strip titles, sort tokens."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)  # remove punctuation
    tokens = name.split()
    tokens = [t for t in tokens if t not in TITLES]  # remove titles
    tokens.sort()  # canonical token order
    return " ".join(tokens)


def token_set_similarity(a: str, b: str) -> float:
    """
    Compare normalized names using multiple strategies, return 0-100 score.
    Handles:
      - token reordering  ("connor sarah" vs "sarah connor")
      - missing middle names / initials
      - typos
    """
    # rapidfuzz token_set_ratio already sorts & intersects token sets
    token_score = fuzz.token_set_ratio(a, b)

    # Jaro-Winkler on the full normalized string (good for typos)
    jw_score = JaroWinkler.similarity(a, b) * 100

    # partial ratio catches one name being a substring of another
    partial_score = fuzz.partial_ratio(a, b)

    return max(token_score, jw_score, partial_score)


# ── Union-Find ────────────────────────────────────────────────────────────────


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ── main function ─────────────────────────────────────────────────────────────


def cluster_names(
    names: list[str],
    threshold: float = 80.0,  # similarity score 0-100
) -> list[list[str]]:
    """
    Cluster names that likely refer to the same person.

    Parameters
    ----------
    names     : raw name strings
    threshold : minimum similarity score (0-100) to consider two names a match

    Returns
    -------
    List of clusters; each cluster is a list of original name strings.
    """
    n = len(names)
    normalized = [normalize_name(name) for name in names]
    uf = UnionFind(n)

    for i, j in combinations(range(n), 2):
        score = token_set_similarity(normalized[i], normalized[j])
        if score >= threshold:
            uf.union(i, j)

    # group original names by cluster root
    clusters: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        root = uf.find(i)
        clusters.setdefault(root, []).append(name)

    return list(clusters.values())


class NameClusterer:
    def num_clusters(self, names: list[str]) -> int:
        names = list(set(name.lower().strip() for name in names))
        return len(cluster_names(names))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("names", type=str)
    args = parser.parse_args()

    names = [name.strip() for name in args.names.split(",")]

    name_clusterer = NameClusterer()
    n = name_clusterer.num_clusters(names)
    print(n)
