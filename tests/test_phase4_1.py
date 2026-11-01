from pathlib import Path

import pytest

from src.near_duplicate_review import cross_split_pairs, resolve_review


def test_finds_cross_split_pairs(tmp_path: Path) -> None:
    near = tmp_path / "near.csv"
    near.write_text("distance,left,right\n1,a.jpg,b.jpg\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_path,class_name,split\n"
        "a.jpg,A,train\n"
        "b.jpg,A,test\n", encoding="utf-8"
    )
    assert len(cross_split_pairs(near, manifest)) == 1


def test_resolution_rejects_incomplete_review(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    review.write_text(
        "pair_id,image_a,image_b,class_a,class_b,split_a,split_b,similarity_score,"
        "reviewer_decision,reviewer_notes,reviewed_by,reviewed_at\n"
        "ND-001,a.jpg,b.jpg,A,A,train,test,1,,,,\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("source_path,class_name,split\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        resolve_review(review, manifest, tmp_path / "out")


def test_resolution_accepts_explicit_decision(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    review.write_text(
        "pair_id,image_a,image_b,class_a,class_b,split_a,split_b,similarity_score,"
        "reviewer_decision,reviewer_notes,reviewed_by,reviewed_at\n"
        "ND-001,a.jpg,b.jpg,A,A,train,test,1,different_apple,kept separate,R,2026-11-01\n",
        encoding="utf-8"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_path,derived_path,class_name,split,sha256,is_valid_image\n"
        "a.jpg,out/a.jpg,A,train,x,True\n"
        "b.jpg,out/b.jpg,A,test,y,True\n", encoding="utf-8"
    )
    result = resolve_review(review, manifest, tmp_path / "out")
    assert result["approved_images"] == 2
