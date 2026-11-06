"""Strict Registry router. Action service and Perception service stay unchanged."""
from __future__ import annotations
import inspect
import math

from plan.full_task.registry import WorkflowValidationError, is_reference


class TargetNotFound(WorkflowValidationError):
    pass


def _target_from_detections(data, target_class):
    """Exact class match. Never substitute another object or synthesize XYZ."""
    if not isinstance(data, dict):
        raise WorkflowValidationError('YOLO response must be a mapping')
    # get_detections_value() was observed to return {'detections':[...]}.
    detections = data.get('detections')
    if not isinstance(detections, list):
        raise WorkflowValidationError('YOLO response missing detections list')
    matching = []
    for item in detections:
        if not isinstance(item, dict) or item.get('class_name') != target_class:
            continue
        xyz = item.get('robot_xyz')
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3 or any(
            type(x) not in (float, int) or not math.isfinite(x) for x in xyz
        ):
            continue
        confidence = item.get('confidence', 0)
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            continue
        matching.append((float(confidence), item))
    if not matching:
        classes = sorted({str(item.get('class_name')) for item in detections if isinstance(item, dict)})
        raise TargetNotFound(f"target_not_found or invalid_xyz: target={target_class!r}; detected={classes}")
    matching.sort(key=lambda pair: pair[0], reverse=True)
    best = matching[0][1]
    return {'target': {'class_name': best['class_name'],
                       'confidence': float(best.get('confidence', 0)),
                       'robot_xyz': [float(x) for x in best['robot_xyz']]},
            'detection_count': len(detections),
            'matching_count': len(matching)}


def expected_tool_routes(plan, catalog):
    """Read-only routing preview. This is metadata, NOT an execution outcome."""
    rows = []
    for step in plan['steps']:
        name = step['tool_name']
        module = catalog[name]['module']
        if module == 'action':
            service = f'action_service.{name}'
        elif module == 'perception' and name == 'get_detections_value':
            service = 'perception_service.get_detections_value'
        else:
            service = f'unsupported_module.{name}'
        rows.append({'step_id': step['step_id'], 'tool_name': name,
                     'module': module, 'service_function': service,
                     'has_deferred_arguments': any(is_reference(v) for v in step['arguments'].values())})
    return rows


def preflight_service_names(plan, catalog):
    """Read-only service existence/signature check BEFORE any physical dispatch.

    Both Action and Perception are checked, so missing YOLO dependencies cannot
    be discovered *after* the first physical Action has moved a robot.
    This does NOT inspect robot readiness, motion envelopes, or collisions.
    """
    from services import action_service, camera_service, perception_service
    missing = []
    verified = []
    for step in plan['steps']:
        name = step['tool_name']
        module = catalog[name]['module']
        if module == 'action':
            func = getattr(action_service, name, None)
            kwargs = dict(step['arguments'])
            for key, value in kwargs.items():
                if is_reference(value):
                    kwargs[key] = 0.0 if catalog[name]['parameters'][key].get('type') == 'number' else ''
        elif module == 'perception' and name == 'get_detections_value':
            func = getattr(perception_service, 'get_detections_value', None)
            kwargs = {'camera_name': step['arguments']['camera_name'],
                      'draw': False, 'include_robot_xyz': True}
            camera_start = getattr(camera_service, 'start_camera', None)
            if not callable(camera_start):
                missing.append('camera_service.start_camera() missing')
            else:
                try:
                    inspect.signature(camera_start).bind(camera_name=step['arguments']['camera_name'])
                except (TypeError, ValueError) as exc:
                    missing.append(f'camera_service.start_camera() signature mismatch: {exc}')
        else:
            missing.append(f'Unsupported tool module/adapter: {module}.{name}')
            continue
        if not callable(func):
            missing.append(f'{module}.{name}() missing')
            continue
        try:
            signature = inspect.signature(func)
            signature.bind(**kwargs)
        except (TypeError, ValueError) as exc:
            missing.append(f'{module}.{name}() signature mismatch: {exc}')
            continue
        verified.append(f'{module}.{name}')
    if missing:
        raise WorkflowValidationError('Service/Registry mismatch (no motion dispatched): ' + '; '.join(sorted(set(missing))))
    return {'passed': True, 'service_invoked': False,
            'robot_motion_dispatched': False, 'checked_tools': sorted(set(verified)),
            'motion_safety_checked': False}


def _real_action_execute(plan, step_id, name, arguments, *, allow_operator_attestation=False):
    """Reuses EXISTING per-step single-use authorization, lock and Action adapter."""
    from plan.adapters.agent_action import inspect_real_action_step
    from agent.task_runner import (
        _agent_auto_real_request, authorize_autonomous_plan, execute_prepared_plan,
    )
    action_step = {'step_id': step_id, 'action_name': name,
                   'arguments': arguments, 'status': 'planned'}
    check = inspect_real_action_step(action_step)
    if check.get('gateway_preflight_passed') is not True:
        raise WorkflowValidationError('Action preflight failed: ' + str(check.get('error')))
    single = {'plan_id': plan['plan_id'] + '_' + step_id,
              'user_text': plan['user_text'], 'planner': 'workflow_full_v1',
              'steps': [action_step], 'worldmodel': {'used': False},
              'mode': 'mock', 'physical_execution_allowed': False}
    with _agent_auto_real_request():
        token = authorize_autonomous_plan(single)
        report = execute_prepared_plan(single, mode='real', approval_token=token)
    rows = report.get('steps')
    status = rows[0]['status'] if isinstance(rows, list) and len(rows) == 1 else None
    if status == 'real_completed_verified':
        return {'result': True, 'service_status': status, 'report': report}
    # WORKFLOW_AUTO_DISPATCH_V1: a confirmed successful dispatch is not
    # verified physical completion. check=true gates the next step;
    # check=false continues after the service call returns.
    if status == 'real_dispatched_unverified':
        execution = rows[0].get('execution') if isinstance(rows, list) and len(rows) == 1 else None
        if not isinstance(execution, dict) or execution.get('result') is not True or execution.get('simulated') is not False:
            raise WorkflowValidationError(f'{name}: dispatch not confirmed successful; stop')
        if execution.get('physical_completion_verified') is True:
            raise WorkflowValidationError(f'{name}: inconsistent completion status; stop')
        raw = execution.get('service_response')
        if isinstance(raw, dict):
            if raw.get('result') is False or raw.get('action_converged') is False:
                raise WorkflowValidationError(f'{name}: Service explicitly did not complete; stop')
            if raw.get('status') in ('max_batches_reached', 'failed', 'error'):
                raise WorkflowValidationError(f'{name}: Service reported failure; stop')
        return {'result': True, 'service_status': status,
                'pending_operator_verification': bool(allow_operator_attestation),
                'physical_completion_verified': False, 'report': report}
    # raise WorkflowValidationError(f'{name}: physical action not verified ({status}); stop')


class ToolRouter:
    def __init__(self, *, mode='mock', mock_detections=None, allow_operator_attestation=False):
        if mode not in ('mock', 'real', 'perception-test'):
            raise ValueError('mode must be mock/real/perception-test')
        self.mode = mode
        self.mock_detections = mock_detections
        self.allow_operator_attestation = bool(allow_operator_attestation)

    def dispatch(self, plan, step, arguments, catalog):
        name = step['tool_name']
        module = catalog[name]['module']
        if module == 'action':
            if self.mode in ('mock', 'perception-test'):
                return {'result': True, 'simulated': True, 'action_name': name,
                        'arguments': dict(arguments)}
            return _real_action_execute(plan, step['step_id'], name, arguments,
                                        allow_operator_attestation=self.allow_operator_attestation)
        if module != 'perception':
            raise WorkflowValidationError(f'Unsupported tool module: {module}')
        if name != 'get_detections_value':
            raise WorkflowValidationError(f'Unsupported Perception adapter: {name}')
        if self.mode == 'mock':
            if self.mock_detections is None:
                raise WorkflowValidationError('Mock YOLO requires --mock-detections fixture; no invented observations')
            data = self.mock_detections
        else:
            from services import camera_service, perception_service
            camera = arguments['camera_name']
            started = camera_service.start_camera(camera)
            if not isinstance(started, dict) or started.get('result') is not True:
                raise WorkflowValidationError('Camera start failed: ' + repr(started))
            data = perception_service.get_detections_value(
                camera_name=camera, draw=False, include_robot_xyz=True)
        return {'result': True, 'simulated': self.mode == 'mock',
                **_target_from_detections(data, arguments['target_class'])}
