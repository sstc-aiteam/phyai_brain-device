"""Optional bounded failure replanning: no automatic replay of real motion."""
from __future__ import annotations
from agent.workflow.runner import run_workflow
from agent.workflow.planner import plan_full_task


def run_with_replan(plan, *, mode='mock', mock_detections=None,
                    provider='local', camera_name='left', model=None,
                    max_replans=1):
    """Mock: replan on ordinary tool failure, not when YOLO target is absent.

    Real: no unverified physical outcome may be replayed or silently rerouted.
    This MVP stops for operator inspection rather than automatically replaying
    partially executed real robot tasks.
    """
    if not isinstance(max_replans, int) or not 0 <= max_replans <= 2:
        raise ValueError('max_replans must be 0..2')
    result = run_workflow(plan, mode=mode, mock_detections=mock_detections)
    attempts = [{'plan': plan, 'result': result}]
    original_text = plan['user_text']
    for count in range(max_replans):
        if result['success'] or result.get('stopped_reason') == 'target_not_found':
            break
        if mode == 'real':
            result['replan_status'] = 'requires_operator_review_after_physical_failure'
            break
        history = [{'tool_name': row['tool_name'], 'arguments': row.get('arguments'),
                    'status': row['status'], 'error': row.get('error')}
                   for attempt in attempts for row in attempt['result'].get('records', [])]
        revised = plan_full_task(original_text, provider=provider,
                                 camera_name=camera_name, model=model,
                                 failure_context={
                                     'attempts': count + 1,
                                     'executed_steps': history,
                                     'original_goal': original_text,
                                 })
        result = run_workflow(revised, mode=mode, mock_detections=mock_detections)
        attempts.append({'plan': revised, 'result': result})
    return {'success': result['success'], 'attempts': attempts,
            'replans': len(attempts)-1, 'final': result}
