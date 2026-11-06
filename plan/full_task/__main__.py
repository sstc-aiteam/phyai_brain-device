"""New opt-in entrypoint: python -m agent.workflow (legacy Auto untouched)."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from plan.full_task.registry import all_tools, validate_plan
from plan.full_task.planner import plan_full_task
from plan.full_task.orchestrator import run_with_replan
from plan.full_task.router import expected_tool_routes, preflight_service_names
from plan.full_task.runner import run_workflow


def _print(label, value):
    print(f'\n=== {label} ===')
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _operator_step_control(*, step, arguments, previous_step, previous_arguments,
                           retry_available, mode, phase='before', previous_status=None):
    if phase == 'final':
        _print('FINAL STEP - OPERATOR CONFIRMATION', {
            'step_id': step['step_id'], 'tool_name': step['tool_name'],
            'arguments': arguments, 'status': previous_status or 'service_verified',
        })
        print('FINAL c: 我已確認最後一步完成；結束 Workflow。')
    else:
        _print('NEXT STEP', {
            'step_id': step['step_id'], 'tool_name': step['tool_name'],
            'arguments': arguments, 'mode': mode,
        })
        if previous_step is None:
            print('FIRST c: 確認實機已就緒，執行第一步（此時尚無前一步可驗證）。')
        elif previous_status == 'awaiting_operator_verification':
            print(f"PREVIOUS {previous_step['step_id']} ({previous_step['tool_name']}): "
                  'Service 已派送但沒有完成證據。')
            print('c = 我已目視確認上一個步驟完成、目前狀態安全，並授權執行這一步。')
            print('NOTE: 人工驗證將另記為 completed_operator_verified，'
                  '不會竄改 Service 的 task_completed_verified。')
        else:
            print(f"Previous step {previous_step['step_id']} Service/test verified.")
            print('c = 確認上一個步驟已完成，並授權執行這一步。')
    while True:
        prompt = '[c] 完成＋驗證並繼續 / [q] 中止'
        if retry_available:
            prompt += ' / [r] 再一次上一步'
        print('NOTE: 確認物體狀態、夾爪、路徑無障礙；此確認不等於感測器驗證。')
        answer = input(prompt + ' > ').strip().lower()
        if answer in ('c', 'continue'):
            return 'continue'
        if answer in ('q', 'quit', 'stop'):
            return 'stop'
        if answer in ('r', 'retry') and retry_available:
            last = previous_step
            _print('PREVIOUS STEP REPEAT REQUEST', {
                'step_id': last['step_id'],
                'tool_name': last['tool_name'],
                'arguments': previous_arguments,
                'warning': 'RE-EXECUTE, NOT UNDO; a previous movement is not rolled back.',
            })
            if mode == 'real' and last['tool_name'] != 'get_detections_value':
                print('Before repeating a physical Action, restore its PRE-action state')
                print('and verify robot clearance; repeated grabs/door motions may collide.')
                expected = f'RESET CONFIRMED {last["step_id"]}'
                if input(f'Type exactly {expected} to re-dispatch > ').strip() != expected:
                    print('Repeat cancelled; no action sent.')
                    continue
            else:
                expected = f'REPEAT {last["step_id"]}'
                if input(f'Type exactly {expected} > ').strip() != expected:
                    print('Repeat cancelled; no action sent.')
                    continue
            return 'repeat_previous'
        print('Invalid choice or repeat not available. No action sent.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent full-plan Workflow MVP')
    parser.add_argument('--text', help='User task')
    parser.add_argument('--plan-file', help='Offline plan JSON (skip LLM for testing)')
    parser.add_argument('--provider', choices=('local','openai'), default='local')
    parser.add_argument('--camera', choices=('left','right','middle'), default='left')
    parser.add_argument('--model')
    parser.add_argument('--mode', choices=('plan','check','mock','perception-test','real'), default='plan')
    parser.add_argument('--replans', type=int, choices=(0,1,2), default=1, help='bounded replan only in mock mode; YOLO target missing always stops')
    parser.add_argument('--step-control', action='store_true', help='Operator c confirms previous step success AND authorizes next; final c confirms last step')
    parser.add_argument('--mock-detections', help='JSON fixture with {"detections":[...]} for mock mode')
    args = parser.parse_args(argv)
    if bool(args.text) == bool(args.plan_file):
        parser.error('Supply exactly one of --text or --plan-file')
    if args.mock_detections and args.mode != 'mock':
        parser.error('--mock-detections only applies to --mode mock')
    if args.step_control and args.mode not in ('mock', 'perception-test', 'real'):
        parser.error('--step-control requires mock, perception-test, or real mode')
    if args.step_control and not sys.stdin.isatty():
        parser.error('--step-control requires an interactive TTY')
    try:
        if args.plan_file:
            plan = validate_plan(json.loads(Path(args.plan_file).read_text(encoding='utf-8')))
        else:
            from agent.capability_selector import select_capability
            capability = select_capability(args.text, provider=args.provider, model=args.model)
            _print('AGENT CAPABILITY', capability)
            if capability.get('tool_name') != 'plan_task' or capability.get('arguments') != {}:
                raise RuntimeError('Workflow CLI expects task capability plan_task; use legacy Agent for information-only queries')
            plan = plan_full_task(args.text, provider=args.provider,
                                  camera_name=args.camera, model=args.model)
        plan.setdefault('plan_id', uuid4().hex)
        plan.setdefault('user_text', args.text or '[offline test plan]')
        _print('FULL PLAN', plan)
        catalog = all_tools()
        _print('TOOL ROUTES', expected_tool_routes(plan, catalog))
        if args.mode == 'check':
            _print('SERVICE PREFLIGHT', preflight_service_names(plan, catalog))
            return 0
        if args.mode == 'plan':
            return 0
        fixture = json.loads(Path(args.mock_detections).read_text(encoding='utf-8')) if args.mock_detections else None
        if args.mode == 'real':
            # Explicit human confirmation of the entire plan. Each physical step
            # also passes through existing one-shot task_runner authorization.
            if os.getenv('AGENT_REAL_ACTION_ENABLED') != '1' or not sys.stdin.isatty():
                raise RuntimeError('Real mode requires AGENT_REAL_ACTION_ENABLED=1 and an interactive terminal')
            expected = 'RUN WORKFLOW ' + plan['plan_id']
            print(f'Confirm entire real workflow by typing: {expected}')
            if input('> ').strip() != expected:
                raise RuntimeError('Real workflow not confirmed')
        if args.step_control:
            # Explicitly disable all replanning while operator-gated.
            result = run_workflow(plan, mode=args.mode, mock_detections=fixture,
                                  step_control=_operator_step_control)
            report = {'success': result['success'],
                      'attempts': [{'plan': plan, 'result': result}],
                      'replans': 0, 'final': result}
        else:
            report = run_with_replan(plan, mode=args.mode, mock_detections=fixture,
                                     provider=args.provider, camera_name=args.camera,
                                     model=args.model, max_replans=args.replans if args.mode == 'mock' else 0)
        _print('WORKFLOW RESULT', report)
        return 0 if report['success'] else 1
    except (Exception, KeyboardInterrupt) as exc:
        _print('WORKFLOW STOPPED', {'error_type': type(exc).__name__, 'message': str(exc)})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
