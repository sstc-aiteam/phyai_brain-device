"""Execute Workflow steps; manual c attests the preceding step, not Service telemetry."""
from __future__ import annotations
from copy import deepcopy

from agent.workflow.registry import (
    all_tools, validate_plan, resolve_arguments, WorkflowValidationError,
)
from agent.workflow.router import ToolRouter, TargetNotFound, preflight_service_names


def _completion_status(mode, module):
    return ('mock_completed' if mode == 'mock' else
            'simulated_action' if mode == 'perception-test' and module == 'action' else
            'perception_observed' if mode == 'perception-test' else
            'completed_verified')


def run_workflow(plan, *, mode='mock', mock_detections=None, step_control=None):
    """Run one full plan, optionally operator-gated at each step boundary.

    At boundary before step N (N>1), 'continue' attests that the previous
    real step was completed when Service returned real_dispatched_unverified,
    AND authorizes dispatching step N. First gate's 'continue' only starts S1.
    After final step, an additional final gate confirms last step's outcome.

    Physical completion verified by a HUMAN is recorded separately from
    Service-verified completion; the Service's telemetry is NEVER rewritten.
    Explicit Service errors/non-convergence never enter manual attestation.
    """
    catalog = all_tools()
    plan = validate_plan(plan, catalog)
    operator_mode = mode == 'real' and step_control is not None
    router = ToolRouter(mode=mode, mock_detections=mock_detections,
                        allow_operator_attestation=operator_mode)
    if mode == 'real':
        preflight_service_names(plan, catalog)
    records = []
    results = {}
    last_arguments = {}
    repeated_step_ids = set()
    # At most one preceding step may be unverified at any point.
    pending = None
    attested_steps = []

    def report(success, reason=None):
        # WORKFLOW_AUTO_DISPATCH_V1: never mislabel dispatch-only as physical completion.
        unverified = [r['step_id'] for r in records
                      if r.get('status') == 'dispatched_unverified']
        out = {'success': success, 'plan_id': plan.get('plan_id'),
               'mode': mode, 'records': records,
               'physical_task_completed': bool(success and mode == 'real' and not unverified)}
        if mode == 'real':
            out['completion_evidence'] = ('incomplete' if not success else
                                          'dispatch_only_unverified' if unverified else
                                          'operator_attested_and_service_verified' if attested_steps else
                                          'service_verified_only')
            out['unverified_steps'] = list(unverified)
            out['operator_attested_steps'] = list(attested_steps)
            out['service_telemetry_preserved'] = True
        if reason is not None:
            out['stopped_reason'] = reason
        elif mode == 'mock':
            out['note'] = 'Mock only; no robot or camera invoked'
        elif mode == 'perception-test':
            out['note'] = 'Perception-test: Actions simulated, real camera/YOLO used; robot NOT moved'
        elif unverified:
            out['note'] = 'Action dispatch accepted; physical completion NOT verified'
        elif attested_steps:
            out['note'] = 'Operator-attested physical steps; NOT independently sensor verified'
        else:
            out['note'] = 'Services returned explicit completion evidence'
        return out

    def execute(step, arguments, *, repeated=False):
        nonlocal pending
        rec = {'step_id': step['step_id'], 'tool_name': step['tool_name'],
               'module': catalog[step['tool_name']]['module'],
               'status': 'running', 'arguments': deepcopy(arguments)}
        if repeated:
            rec['repeat_of'] = step['step_id']
            rec['repeat_attempt'] = 2
        records.append(rec)
        try:
            execution = router.dispatch(plan, step, arguments, catalog)
            if not isinstance(execution, dict) or execution.get('result') is not True:
                raise WorkflowValidationError('Tool did not report success')
            results[step['step_id']] = execution
            last_arguments[step['step_id']] = deepcopy(arguments)
            rec['result'] = execution
            if operator_mode and execution.get('pending_operator_verification') is True:
                rec['status'] = 'awaiting_operator_verification'
                rec['verification'] = {'method': 'pending_operator',
                                       'service_completion_verified': False}
                pending = rec
            elif (mode == 'real' and rec['module'] == 'action'
                  and execution.get('service_status') == 'real_dispatched_unverified'):
                rec['status'] = 'dispatched_unverified'
                rec['verification'] = {'method': 'service_dispatch_only',
                                       'service_completion_verified': False}
                pending = None
            else:
                rec['status'] = _completion_status(mode, rec['module'])
                rec['verification'] = {'method': 'service' if mode == 'real' else 'test',
                                       'service_completion_verified': mode == 'real'}
                pending = None
            return None
        except TargetNotFound as exc:
            rec['status'] = 'blocked'
            rec['error'] = str(exc)
            return report(False, 'target_not_found')
        except Exception as exc:
            rec['status'] = 'failed'
            rec['error'] = f'{type(exc).__name__}: {exc}'
            return report(False, 'tool_failure')

    def attest_pending():
        nonlocal pending
        if pending is None:
            return
        pending['status'] = 'completed_operator_verified'
        pending['verification'] = {
            'method': 'operator_attestation',
            'service_completion_verified': False,
            'operator_input': 'c',
        }
        attested_steps.append(pending['step_id'])
        pending = None

    def gate(step, arguments, previous, *, phase):
        previous_args = last_arguments.get(previous['step_id']) if previous else None
        can_repeat = (previous is not None
                      and previous['step_id'] in results
                      and previous['step_id'] not in repeated_step_ids)
        while True:
            try:
                choice = step_control(
                    step=step, arguments=deepcopy(arguments),
                    previous_step=previous,
                    previous_arguments=deepcopy(previous_args),
                    retry_available=can_repeat, mode=mode, phase=phase,
                    previous_status=pending['status'] if pending is not None else None)
            except (EOFError, KeyboardInterrupt):
                choice = 'stop'
            except Exception as exc:
                records.append({'step_id': step['step_id'],
                                'tool_name': step['tool_name'],
                                'module': catalog[step['tool_name']]['module'],
                                'status': 'failed',
                                'error': f'step_control_error: {type(exc).__name__}: {exc}'})
                return 'stop', report(False, 'step_control_error')
            if choice == 'stop':
                records.append({'step_id': step['step_id'],
                                'tool_name': step['tool_name'],
                                'module': catalog[step['tool_name']]['module'],
                                'status': 'operator_stopped',
                                'arguments': deepcopy(arguments), 'phase': phase})
                return 'stop', report(False, 'operator_stopped')
            if choice == 'repeat_previous':
                if not can_repeat:
                    records.append({'step_id': step['step_id'],
                                    'tool_name': step['tool_name'],
                                    'module': catalog[step['tool_name']]['module'],
                                    'status': 'blocked',
                                    'error': 'No preceding eligible step available for repeat'})
                    return 'stop', report(False, 'unsafe_repeat_blocked')
                repeated_step_ids.add(previous['step_id'])
                # DO NOT mark old pending as operator verified. Superseded by
                # explicitly authorized repeat (the operator reset pre-state).
                if pending is not None:
                    pending['status'] = 'superseded_by_operator_repeat'
                failure = execute(previous, previous_args, repeated=True)
                if failure is not None:
                    return 'stop', failure
                return 'repeat_previous', None
            if choice == 'continue':
                attest_pending()
                return 'continue', None
            records.append({'step_id': step['step_id'],
                            'tool_name': step['tool_name'],
                            'module': catalog[step['tool_name']]['module'],
                            'status': 'failed',
                            'error': f'Invalid operator choice: {choice!r}'})
            return 'stop', report(False, 'invalid_operator_choice')

    for index, step in enumerate(plan['steps']):
        previous = plan['steps'][index - 1] if index else None
        while True:
            try:
                args = resolve_arguments(step, results, catalog)
            except Exception as exc:
                records.append({'step_id': step['step_id'], 'tool_name': step['tool_name'],
                                'module': catalog[step['tool_name']]['module'],
                                'status': 'failed',
                                'error': f'{type(exc).__name__}: {exc}'})
                return report(False, 'argument_resolution_failed')
            if step_control is not None:
                choice, failure = gate(step, args, previous, phase='before')
                if failure is not None:
                    return failure
                if choice == 'repeat_previous':
                    # A repeated YOLO may update S4's deferred robot_xyz.
                    continue
            failure = execute(step, args)
            if failure is not None:
                return failure
            break

    if step_control is not None:
        last = plan['steps'][-1]
        while True:
            choice, failure = gate(last, last_arguments[last['step_id']], last, phase='final')
            if failure is not None:
                return failure
            if choice == 'continue':
                break
            # Final r explicitly redispatches LAST step; ask to attest it anew.

    # No unconfirmed physical dispatch may yield a successful real report.
    if pending is not None:
        return report(False, 'operator_verification_missing')
    return report(True)
