import torch

from sdm.data.neucodec_dataset import (
    CLS_ID,
    MASK_ID,
    NUM_SPECIAL,
    PAD_ID,
    SEP_ID,
    codes_to_input_ids,
    collate,
)


def test_special_token_ids_distinct():
    assert {PAD_ID, CLS_ID, SEP_ID, MASK_ID} == {0, 1, 2, 3}
    assert NUM_SPECIAL == 4


def test_codes_to_input_ids_short_sequence():
    codes = [10, 20, 30]
    ids, mask = codes_to_input_ids(codes, max_length=8)
    assert ids.tolist() == [CLS_ID, 14, 24, 34, SEP_ID, PAD_ID, PAD_ID, PAD_ID]
    assert mask.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def test_codes_to_input_ids_truncation():
    codes = list(range(100))
    ids, mask = codes_to_input_ids(codes, max_length=8)
    assert ids[0].item() == CLS_ID
    assert ids[-1 - mask.tolist()[::-1].index(1)].item() == SEP_ID
    assert mask.sum().item() == 8


def test_codes_to_input_ids_no_specials():
    codes = [1, 2, 3]
    ids, mask = codes_to_input_ids(codes, max_length=4, add_special_tokens=False)
    assert ids.tolist() == [5, 6, 7, PAD_ID]
    assert mask.tolist() == [1, 1, 1, 0]


def test_collate_stacks_batch():
    a, am = codes_to_input_ids([1, 2], max_length=4)
    b, bm = codes_to_input_ids([5, 6, 7], max_length=4)
    batch = collate(
        [
            {"input_ids": a, "attention_mask": am, "id": "x"},
            {"input_ids": b, "attention_mask": bm, "id": "y"},
        ]
    )
    assert batch["input_ids"].shape == (2, 4)
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["ids"] == ["x", "y"]
