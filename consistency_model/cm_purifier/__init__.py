"""Pixel-space consistency purifier for targeted poisoning defenses."""

ATTACK_TO_ID = {"clean": 0, "wb": 1, "bp": 2}
ID_TO_ATTACK = {value: key for key, value in ATTACK_TO_ID.items()}

