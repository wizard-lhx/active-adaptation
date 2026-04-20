"""
Utilities for unifying the API of different backends.
"""


def find_sensor_bodies(contact_sensor, body_names: str | list[str]) -> tuple[list[int], list[str]]:
    try:
        body_ids, body_names = contact_sensor.find_bodies(body_names)
    except AttributeError:
        from mjlab.utils.lab_api.string import resolve_matching_names
        body_ids, body_names = resolve_matching_names(body_names, contact_sensor.primary_names)
    return body_ids, body_names

