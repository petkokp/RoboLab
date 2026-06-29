# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg

# --- isaacsim6/isaaclab3 compat shim: contact-sensor backend guard (upstream strictness) ---
# WHAT CRASHES (exact, verified by reverting this guard):
#   CRASHTEST GrabAFruitTask: CRASH[create_env] AttributeError:
#   'NoneType' object has no attribute 'filter_count'
# WHERE (isaaclab_physx/sensors/contact_sensor/contact_sensor.py):
#   _initialize_impl (:324) calls physics_sim_view.create_rigid_contact_view(body, filter_patterns=...);
#   when a FILTER target body has no contact-capable collider, PhysX returns a view whose
#   ._backend is None. Then _create_buffers (:357) does
#     self._num_filter_shapes = self.contact_view.filter_count if self.cfg.filter_prim_paths_expr else 0
#   and `filter_count` dereferences the None backend -> AttributeError, aborting scene setup.
# WHY RoboLab hits it: create_contact_sensors() (below) builds O(n^2) pairwise gripper-object
#   and object-object sensors across the whole contact_object_list; some scene objects carry no
#   contact-capable collider. isaaclab3 validates that the SENSOR body has PhysxContactReportAPI
#   (_initialize_impl :311 raises otherwise) but does NOT validate the FILTER targets, so a
#   no-collider filter crashes instead of reporting zero contact -- this is genuinely stricter
#   than the pre-upgrade (isaacsim4) behaviour, not a renamed API.
# FIX: probe contact_view.filter_count at _create_buffers time (the definitive moment PhysX has
#   decided); if it's backend-less, drop the filters so the sensor initialises INERT (the
#   downstream get_contact_force/in_contact in world_state.py treat filter_prim_paths_expr=None
#   as zero-contact). Reacting to the real backend state is why this is preferred over pre-
#   filtering sensors by guessing which objects PhysX will give a backend (see PR discussion).
try:
    from isaaclab_physx.sensors.contact_sensor.contact_sensor import (
        ContactSensor as _ISCS,
    )

    if not getattr(_ISCS, "_robolab_backend_guard", False):
        _orig_create_buffers = _ISCS._create_buffers

        def _guarded_create_buffers(self):
            try:
                if self.cfg.filter_prim_paths_expr:
                    _ = self.contact_view.filter_count  # probe the PhysX backend
            except AttributeError:
                print(
                    f"[robolab/contact] no contact-view backend for "
                    f"'{self.cfg.prim_path}' (filters={self.cfg.filter_prim_paths_expr}); "
                    f"initializing this sensor WITHOUT filters (inert)",
                    flush=True,
                )
                self.cfg.filter_prim_paths_expr = None
            return _orig_create_buffers(self)

        _ISCS._create_buffers = _guarded_create_buffers
        _ISCS._robolab_backend_guard = True
except Exception as _e:  # pragma: no cover - defensive
    print(f"[robolab/contact] could not install contact-sensor backend guard: {_e}")


def create_contact_sensor_cfg(entity_1, entity_2, update_period=0.0, history_length=6, debug_vis=False):
        return ContactSensorCfg(
            prim_path=entity_1,
            update_period=update_period,
            history_length=history_length,
            debug_vis=debug_vis,
            filter_prim_paths_expr=[entity_2],
        )


def create_batch_contact_sensor_cfg(entity_prim_path: str, filter_prim_paths: list[str],
                                     update_period=0.0, history_length=6, debug_vis=False):
    """
    Create a contact sensor that monitors contacts between one entity and multiple filter entities.

    This is more efficient than creating individual pairwise sensors when you need to check
    one body (e.g., gripper) against many objects at once.

    Args:
        entity_prim_path: The prim path of the body to monitor contacts for
        filter_prim_paths: List of prim paths to check contacts against
        update_period: Sensor update period
        history_length: Number of contact history frames to store
        debug_vis: Enable debug visualization

    Returns:
        ContactSensorCfg configured for batch contact detection
    """
    return ContactSensorCfg(
        prim_path=entity_prim_path,
        update_period=update_period,
        history_length=history_length,
        debug_vis=debug_vis,
        filter_prim_paths_expr=filter_prim_paths,
    )


def create_contact_sensors(env_cfg):
    """
    Dynamically create contact sensors based on contact_gripper and contact_object_list.

    Creates:
    1. Batch sensors for each gripper (gripper vs all objects) - named "{gripper_name}__batch"
    2. Individual pairwise sensors for gripper-object pairs (for backwards compatibility)
    3. Individual pairwise sensors for object-object pairs

    Args:
        env_cfg: Environment configuration containing scene, contact_gripper, and contact_object_list
    """

    scene = env_cfg.scene
    if env_cfg.contact_object_list is None or env_cfg.contact_gripper is None:
        return

    # Objects to exclude from batch sensor (these are checked separately or not relevant for wrong-grab detection)
    batch_sensor_exclude = {"table"}

    for gripper_name, gripper_prim_path in env_cfg.contact_gripper.items():
        # Create batch sensor for gripper vs all objects except excluded ones (for efficient batch queries)
        all_object_prim_paths = [getattr(scene, obj_name).prim_path for obj_name in env_cfg.contact_object_list if obj_name not in batch_sensor_exclude]
        batch_sensor_name = f"{gripper_name}__all_objs"
        batch_sensor = create_batch_contact_sensor_cfg(gripper_prim_path, all_object_prim_paths)
        setattr(scene, batch_sensor_name, batch_sensor)

        # Also create individual pairwise sensors (for backwards compatibility with in_contact)
        for obj_name in env_cfg.contact_object_list:
            contact_sensor_name = f"{gripper_name}__{obj_name}"
            contact_sensor = create_contact_sensor_cfg(gripper_prim_path, getattr(scene, obj_name).prim_path)
            setattr(scene, contact_sensor_name, contact_sensor)

    objects = env_cfg.contact_object_list
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj_name = objects[i]
            other_obj_name = objects[j]
            contact_sensor_name = f"{obj_name}__{other_obj_name}"
            contact_sensor = create_contact_sensor_cfg(
                getattr(scene, obj_name).prim_path,
                getattr(scene, other_obj_name).prim_path)
            setattr(scene, contact_sensor_name, contact_sensor)

def get_contact_sensors(scene):
    """
    Get all contact sensors.

    Example:
        env = create_env(...)
        contact_sensors = get_contact_sensors(env.scene)

    Args:
        scene (InteractiveScene): The scene to get the contact sensors from.
    """
    contact_sensors = {
        name: sensor for name, sensor in scene.sensors.items()
        if isinstance(sensor, ContactSensor)
        or sensor.__class__.__name__ == "ContactSensor"
        or (("__" in name or name.endswith("__all_objs")) and hasattr(sensor, "data"))
    }
    return contact_sensors

def get_contact_sensor(scene, body1, body2) -> ContactSensor:
    contact_sensor_string = f"{body1}__{body2}"
    contact_sensor_string2 = f"{body2}__{body1}"

    sensors = get_contact_sensors(scene)
    if contact_sensor_string in sensors.keys():
        return sensors[contact_sensor_string]
    elif contact_sensor_string2 in sensors.keys():
        return sensors[contact_sensor_string2]
    else:
        raise ValueError(f"Contact sensor {contact_sensor_string} or {contact_sensor_string2} not found. available sensors: {sensors.keys()}")


def get_contact_sensor_with_order(scene, body1, body2) -> tuple[ContactSensor, bool]:
    """
    Get the contact sensor between two bodies and whether the order is reversed.

    Args:
        scene: The scene containing sensors
        body1: First body name (desired sensor body - forces on this body)
        body2: Second body name (desired filter body - forces from this body)

    Returns:
        tuple: (ContactSensor, is_reversed)
            - ContactSensor: The contact sensor
            - is_reversed: True if sensor was found as body2__body1 (order reversed),
                           meaning force_matrix_w gives forces on body2 from body1,
                           so the force needs to be negated to get forces on body1.
    """
    contact_sensor_string = f"{body1}__{body2}"
    contact_sensor_string2 = f"{body2}__{body1}"

    sensors = get_contact_sensors(scene)
    if contact_sensor_string in sensors.keys():
        return sensors[contact_sensor_string], False
    elif contact_sensor_string2 in sensors.keys():
        return sensors[contact_sensor_string2], True
    else:
        raise ValueError(f"Contact sensor {contact_sensor_string} or {contact_sensor_string2} not found. available sensors: {sensors.keys()}")


def get_batch_contact_sensor(scene, body: str) -> ContactSensor | None:
    """
    Get the batch contact sensor for a body (e.g., gripper).

    Args:
        scene: The scene containing sensors
        body: Name of the body (e.g., "gripper")

    Returns:
        The batch ContactSensor if it exists, None otherwise
    """
    batch_sensor_name = f"{body}__all_objs"
    sensors = get_contact_sensors(scene)
    return sensors.get(batch_sensor_name)
