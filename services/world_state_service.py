"""Live arm/vision data -> canonical runtime WorldState."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
import config

from agent.runtime.world_state import WorldState
from services import arm_service, task_perception_service, vision_service
from utils.object_identity import assign_request_object_ids


def get_robot_status():
    try:
        response = arm_service.get_arm_status(); data = response.get("data") or {}
        rows = data.get("arms")
        if response.get("result") is not True or not isinstance(rows, list):
            raise RuntimeError(response.get("message") or "arm status unavailable")
        arms = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("arm_name"): continue
            status = row.get("status") or {}; name = str(row["arm_name"])
            item = {
                "arm_name":name, "device_id":f"{name}_arm", "driver":row.get("driver"),
                "connected":status.get("connected") is True,
                "pose":deepcopy(status.get("pose")), "joints":deepcopy(status.get("joints")),
            }
            for f in ("holding","controlling","location"):
                if f in status: item[f] = deepcopy(status[f])
                elif f in row: item[f] = deepcopy(row[f])
            arms.append(item)
        return {"status_available":True,"connected":any(x["connected"] for x in arms),"arms":arms}
    except Exception as exc:
        return {"status_available":False,"connected":False,"arms":[],
                "status_error":f"{type(exc).__name__}: {exc}"}


def get_detected_objects(camera_names=("left","right")):
    raw, timestamps, errors = [], {}, {}
    for camera in camera_names:
        try:
            response = vision_service.get_detections(camera_name=camera, include_robot_xyz=True)
            data = response.get("data") or {}; rows = data.get("detections")
            if response.get("result") is not True or not isinstance(rows, list):
                raise RuntimeError(response.get("message") or "detections unavailable")
            timestamps[camera] = data.get("timestamp")
            raw += [{**deepcopy(x),"camera_source":camera} for x in rows if isinstance(x,dict)]
        except Exception as exc:
            errors[camera] = f"{type(exc).__name__}: {exc}"
    objects = assign_request_object_ids(raw)
    return {
        "detection_available":bool(timestamps), "objects":objects, "count":len(objects),
        "latest_time":timestamps.get("left") or timestamps.get("right"),
        "camera_timestamps":timestamps, "camera_errors":errors,
    }


def build_world_state(*, context: dict[str,Any] | None=None,
                      detected_objects=None, robot_status=None,
                      read_live_robot=True, read_live_vision=True):
    context = deepcopy(context or {})
    world = WorldState({"entities":{},"meta":{
        "source":"services.world_state_service",
        "captured_at":datetime.now(timezone.utc).isoformat(),
    }})
    static = deepcopy(getattr(config,"WORLD_STATIC_ENTITIES",{}) or {})
    if isinstance(context.get("static_entities"),dict): static.update(context["static_entities"])
    for entity, values in static.items():
        if isinstance(values,dict): world.update_entity(str(entity),values)

    robot = robot_status or context.get("robot_status") or (get_robot_status() if read_live_robot else {})
    for arm in robot.get("arms") or []:
        did = str(arm.get("device_id") or "")
        if not did: continue
        values = {"type":"robot_arm","tags":["manipulator"],"arm_name":arm.get("arm_name"),
                  "driver":arm.get("driver"),"connected":arm.get("connected") is True,
                  "pose":deepcopy(arm.get("pose")),"joints":deepcopy(arm.get("joints"))}
        for f in ("holding","controlling","location"):
            if f in arm: values[f]=deepcopy(arm[f])
        world.update_entity(did, values)

    detected = detected_objects or context.get("detected_objects") or (get_detected_objects() if read_live_vision else {})
    for obj in detected.get("objects") or []:
        oid = str(obj.get("object_id") or "")
        if not oid: continue
        cls = str(obj.get("class_name") or "object")
        values = {"type":"object","class_name":cls,"tags":["perceived_object","pickable",cls],
                  "exists":True,"camera_source":obj.get("camera_source"),
                  "confidence":obj.get("confidence"),"yaw_deg":obj.get("yaw_deg")}
        xyz = obj.get("robot_xyz")
        if isinstance(xyz,(list,tuple)) and len(xyz)==3: values["position"]=[float(v) for v in xyz]
        if isinstance(obj.get("reachable"),bool): values["reachable"]=obj["reachable"]
        if isinstance(obj.get("reachable_by"),list): values["reachable_by"]=deepcopy(obj["reachable_by"])
        if isinstance(obj.get("state_tags"),dict): values.update(deepcopy(obj["state_tags"]))
        world.update_entity(oid, values)

    for fact in context.get("task_facts") or []:
        if isinstance(fact,dict) and fact.get("subject") and (fact.get("predicate") or fact.get("field")) and "value" in fact:
            world.set(str(fact["subject"]), str(fact.get("predicate") or fact["field"]), deepcopy(fact["value"]))
    return world


def refresh_fact(world, *, subject, predicate, context=None, camera_source=None):
    row = task_perception_service.observe_fact(
        subject=subject,predicate=predicate,context=context,camera_source=camera_source)
    if row.get("status") != "unresolved" and "value" in row:
        world.set(subject,predicate,deepcopy(row["value"]))
    return row


def refresh_facts(world, requests, *, context=None):
    rows = task_perception_service.observe_facts(requests, context=context)
    for row in rows:
        if row.get("status") != "unresolved" and "value" in row:
            world.set(str(row["subject"]),str(row["predicate"]),deepcopy(row["value"]))
    return rows


def build_live_context(**kwargs):
    robot = kwargs.pop("robot_status",None) or get_robot_status()
    detected = kwargs.pop("detected_objects",None) or get_detected_objects()
    world = build_world_state(robot_status=robot,detected_objects=detected,
                              read_live_robot=False,read_live_vision=False,**kwargs)
    return {"robot_status":robot,"detected_objects":detected,"world_state":world.snapshot()}
