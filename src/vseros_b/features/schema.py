"""Data structures describing feature collections and versions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata describing a single feature column."""

    name: str
    dtype: str = "float32"
    description: str = ""
    version: str = "v1"

    def as_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "version": self.version,
        }


@dataclass
class FeatureSchema:
    """A collection of features that share the same version tag."""

    version: str = "v1"
    features: List[FeatureDefinition] = field(default_factory=list)

    def add(self, feature: FeatureDefinition) -> None:
        if feature.version != self.version:
            raise ValueError(
                f"Feature {feature.name} has version {feature.version}, expected {self.version}."
            )
        if feature.name in {f.name for f in self.features}:
            raise ValueError(f"Feature {feature.name} already registered in schema {self.version}.")
        self.features.append(feature)

    def extend(self, items: Iterable[FeatureDefinition]) -> None:
        for feat in items:
            self.add(feat)

    def select(self, names: Sequence[str]) -> "FeatureSchema":
        subset = [feat for feat in self.features if feat.name in set(names)]
        return FeatureSchema(version=self.version, features=subset)

    def as_dict(self) -> Dict[str, Dict[str, str]]:
        return {feat.name: feat.as_dict() for feat in self.features}

    def names(self) -> List[str]:
        return [feat.name for feat in self.features]
