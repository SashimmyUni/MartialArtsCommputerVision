"""Focused tests for the SQLite candidate catalog."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import candidate_catalog as catalog


class CandidateCatalogTests(unittest.TestCase):
    def test_import_review_and_export_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scout.csv"
            urls = [f"https://www.youtube.com/watch?v=video{i}" for i in range(4)]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["technique", "angle", "video_id", "url", "title", "view_count"],
                )
                writer.writeheader()
                for index, url in enumerate(urls):
                    writer.writerow(
                        {
                            "technique": "Jab",
                            "angle": "front",
                            "video_id": f"video{index}",
                            "url": url,
                            "title": f"Jab {index}",
                            "view_count": str(1000 + index),
                        }
                    )

            connection = catalog.connect(root / "catalog.sqlite")
            try:
                self.assertEqual(catalog.import_scout_csv(connection, source), 4)
                self.assertEqual(catalog.import_scout_csv(connection, source), 4)
                self.assertEqual(len(catalog.list_candidates(connection)), 4)
                for url in urls:
                    self.assertTrue(catalog.review_candidate(connection, "jab", "front", url, "approved"))
                rows = catalog.approved_plan_rows(connection)
            finally:
                connection.close()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["technique"], "jab")
            self.assertEqual(rows[0]["angle"], "front")
            self.assertEqual([rows[0][f"source_url_{i}"] for i in range(1, 5)], list(reversed(urls)))


if __name__ == "__main__":
    unittest.main()
