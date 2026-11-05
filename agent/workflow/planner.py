"""One-shot planner. Does not call WorldModel or any physical service."""
from __future__ import annotations
import json
import re
from pathlib import Path
from uuid import uuid4
from agent.workflow.registry import prompt_tools, validate_plan, WorkflowValidationError

PROMPT_FILE = Path(__file__).with_name('plan_full_task.md')
OBJECT_LABELS_FILE = Path(__file__).with_name('object_labels.json')


def load_object_labels():
    """Load the independent YOLO class <-> Chinese names mapping (no storage rules)."""
    try:
        data = json.loads(OBJECT_LABELS_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise WorkflowValidationError(
            f'Cannot read object class mapping {OBJECT_LABELS_FILE}: {exc}'
        ) from exc
    if not isinstance(data, dict) or not data:
        raise WorkflowValidationError('object_labels.json must be a non-empty mapping')
    label_owner = {}
    for class_name, labels in data.items():
        if (not isinstance(class_name, str) or not class_name.isidentifier()
                or not isinstance(labels, list) or not labels
                or any(not isinstance(label, str) or not label.strip() for label in labels)):
            raise WorkflowValidationError(f'Invalid object_labels entry: {class_name!r}')
        for label in labels:
            if label in label_owner and label_owner[label] != class_name:
                raise WorkflowValidationError(f'Ambiguous object label: {label!r}')
            label_owner[label] = class_name
    return data


def match_object_labels(user_text, object_labels):
    """Resolve labels in user input; longest overlap wins (e.g. 棉花棒包裝)."""
    candidates = []
    for class_name, labels in object_labels.items():
        for label in [class_name, *labels]:
            for match in re.finditer(re.escape(label), user_text, flags=re.IGNORECASE):
                # An English class name must be a full identifier, not a substring.
                if label.isascii() and (match.start() and
                        (user_text[match.start() - 1].isascii() and (user_text[match.start() - 1].isalnum() or user_text[match.start() - 1] == '_'))):
                    continue
                if label.isascii() and (match.end() < len(user_text) and
                        (user_text[match.end()].isascii() and (user_text[match.end()].isalnum() or user_text[match.end()] == '_'))):
                    continue
                candidates.append((match.start(), match.end(), class_name, label))
    selected = []
    for item in sorted(candidates, key=lambda x: (-(x[1] - x[0]), x[0])):
        if not any(item[0] < old[1] and old[0] < item[1] for old in selected):
            selected.append(item)
    out = []
    seen = set()
    for start, end, class_name, label in sorted(selected):
        if class_name not in seen:
            out.append({'class_name': class_name, 'matched_label': label})
            seen.add(class_name)
    return out


def verify_planned_detection_classes(plan, object_labels, matched_objects):
    """Never silently pick a different YOLO target (e.g. 毛巾 -> 口罩)."""
    registered = set(object_labels)
    matched = {obj['class_name'] for obj in matched_objects}
    for step in plan['steps']:
        if step['tool_name'] != 'get_detections_value':
            continue
        target = step['arguments'].get('target_class')
        if not matched:
            raise WorkflowValidationError(
                'No known YOLO object name found in user request; cannot select '
                'target_class without a verified mapping'
            )
        if target not in registered:
            raise WorkflowValidationError(
                f'Unknown YOLO target_class: {target!r}; update object_labels.json only '
                'after confirming the perception class registry'
            )
        if matched and target not in matched:
            raise WorkflowValidationError(
                f'Object class mismatch: requested {sorted(matched)}; '
                f'planner chose {target!r}'
            )
    return plan


OBJECT_MAPPING_INSTRUCTION = """
Object class mapping is authoritative for the target_class of get_detections_value.
The input includes object_labels (YOLO class_name -> Chinese labels/aliases) and
matched_objects (names actually found in the user's request). When a single
matched object is present, use its exact class_name for get_detections_value.
Never use a different class just because an example task used disposable_mask.
For unknown items, do NOT invent a class or substitute an unrelated object.
This mapping is ONLY for object identity; ignore any storage/destination rule
not requested by the user. A user-specified destination takes precedence.
If pick_object needs XYZ, write reference OBJECTS, e.g.
{"$ref":"S3.target.robot_xyz.0"}, {"$ref":"S3.target.robot_xyz.1"},
{"$ref":"S3.target.robot_xyz.2"} for x, y, z respectively; NEVER use strings
like "$ref.S3.robot_xyz.0.x" or fabricate numeric coordinates.
"""



def _normalize_local_pick_refs(steps):
    """Repair ONLY Local pick_object XYZ refs with an unambiguous '$ref:' prefix.

    A repair requires the correct x/y/z index and an earlier get_detections_value
    step. All other strings remain untouched and are rejected by validate_plan.
    Never invent a target, modify a class name, or resolve robot coordinates.
    """
    if not isinstance(steps, list):
        return steps  # Existing validate_plan still rejects malformed plans.
    prior_detection_steps = set()
    normalized = []
    for step in steps:
        if isinstance(step, dict):
            args = step.get('arguments')
            if step.get('tool_name') == 'pick_object' and isinstance(args, dict):
                changed_args = None
                for axis, index in (('x', '0'), ('y', '1'), ('z', '2')):
                    value = args.get(axis)
                    if not (isinstance(value, str) and value.startswith('$ref:')):
                        continue
                    reference = value[len('$ref:'):]
                    pieces = reference.split('.')
                    if (len(pieces) != 4
                            or pieces[1:] != ['target', 'robot_xyz', index]
                            or pieces[0] not in prior_detection_steps):
                        raise WorkflowValidationError(
                            f'Invalid Local {axis} reference: {value!r}; '
                            'expected an earlier detection step and correct XYZ index'
                        )
                    if changed_args is None:
                        changed_args = dict(args)
                    changed_args[axis] = {'$ref': reference}
                if changed_args is not None:
                    step = dict(step)
                    step['arguments'] = changed_args
            if (step.get('tool_name') == 'get_detections_value'
                    and isinstance(step.get('step_id'), str)):
                prior_detection_steps.add(step['step_id'])
        normalized.append(step)
    return normalized



# === WORKFLOW_CAUSAL_PLAN_GUARD_V1 ===
def _validate_workflow_causal_order(plan, user_text):
    """Validate single-arm gripper occupancy and drawer-storage precedence.

    This is a PLAN-ONLY check, not a claim that hardware actually moved.
    No step mutation or automatic reordering: all $ref IDs stay trustworthy.
    """
    steps = plan['steps']
    holding_by_arm = {}
    for step in steps:
        name = step['tool_name']
        args = step['arguments']
        arm = args.get('arm_name')
        if name == 'pick_object':
            if arm in holding_by_arm:
                raise WorkflowValidationError(
                    f"{step['step_id']} pick_object: arm {arm!r} already holding "
                    f"from {holding_by_arm[arm]}; release the item first")
            holding_by_arm[arm] = step['step_id']
        elif name in ('open_drawer', 'close_drawer') and arm in holding_by_arm:
            raise WorkflowValidationError(
                f"{step['step_id']} {name}: arm {arm!r} still holding an item "
                f"picked at {holding_by_arm[arm]}. The same gripper cannot "
                'operate a drawer while holding a picked object. '
                'For storage into a closed/unknown drawer, open it BEFORE picking.')
        elif name == 'place_object':
            holding_by_arm.pop(arm, None)

    # The user's request defines the destination; never assume every "drawer"
    # mention means the object is destined FOR the drawer.
    text = ''.join(user_text.lower().split())
    store_to_drawer = bool(
        re.search(r'(?:放|塞|收|存|擺|置|移)(?:進|入|到|在|至).{0,12}?(?:抽屜)', text)
        or re.search(r'(?:put|place|store).{0,80}(?:in|into).{0,30}drawer',
                     user_text.lower())
    )
    if not store_to_drawer:
        return plan
    already_open = bool(
        re.search(r'抽屜(?:已|已經|本來|原本|目前|現在|是)(?:經)?(?:打開|開了|開著|開的|打開了)', text)
        or re.search(r'drawer.{0,25}(?:already|currently)\s+open', user_text.lower())
    )
    opens = [i for i, st in enumerate(steps) if st['tool_name'] == 'open_drawer']
    picks = [i for i, st in enumerate(steps) if st['tool_name'] == 'pick_object']
    places = [i for i, st in enumerate(steps) if st['tool_name'] == 'place_object']
    closes = [i for i, st in enumerate(steps) if st['tool_name'] == 'close_drawer']

    if not places:
        raise WorkflowValidationError('Drawer-storage task missing place_object')
    if not picks:
        raise WorkflowValidationError('Drawer-storage task missing pick_object')
    if not already_open:
        # Drawer not explicitly said open => open before holding the object.
        if not opens or opens[0] >= picks[0] or opens[0] >= places[0]:
            raise WorkflowValidationError(
                'Drawer storage: drawer is not explicitly already open; '
                'schedule open_drawer BEFORE pick_object and place_object. '
                'Do not make the user say "open first".')
    if places[0] < picks[0]:
        raise WorkflowValidationError('Drawer storage: place_object before pick_object')
    if closes and closes[0] < places[-1]:
        raise WorkflowValidationError('Drawer storage: close_drawer before place_object')
    return plan

def plan_full_task(user_text, *, provider='local', camera_name='left', model=None, failure_context=None):
    if provider not in ('local', 'openai') or camera_name not in ('left', 'right', 'middle'):
        raise ValueError('Invalid provider or camera_name')
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError('user_text required')
    object_labels = load_object_labels()
    matched_objects = match_object_labels(user_text, object_labels)
    from agent.providers.structured import chat_structured
    schema = {
        'type': 'object', 'properties': {
            'supported': {'type': 'boolean'},
            'reason': {'type': 'string'},
            'steps_json': {'type': 'string'},
        }, 'required': ['supported', 'reason', 'steps_json'],
        'additionalProperties': False,
    }
    payload = {'user_text': user_text, 'camera_name': camera_name,
               'default_arm': camera_name if camera_name in ('left','right') else 'left',
               'tools': prompt_tools(), 'failure_context': failure_context,
               'object_labels': object_labels, 'matched_objects': matched_objects}
    # === WORKFLOW_CAUSAL_REPLAN_V1 ===
    for _causal_attempt in range(2):
        answer = chat_structured(
            stage='agent_workflow_full_plan', provider=provider, model=model,
            messages=[{'role': 'system', 'content': PROMPT_FILE.read_text(encoding='utf-8') + '\n' + OBJECT_MAPPING_INSTRUCTION},
                      {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
            response_schema=schema, temperature=0.0, max_tokens=4096,
            num_ctx=16384, timeout=180.0, owner='agent_workflow_full_plan',
        )
        content = answer.get('content') if isinstance(answer, dict) else None
        result = json.loads(content) if isinstance(content, str) else content
        if not isinstance(result, dict) or set(result) != set(schema['required']) or type(result.get('supported')) is not bool or not isinstance(result.get('reason'), str) or not isinstance(result.get('steps_json'), str):
            raise WorkflowValidationError('Full Planner response schema invalid')
        if result['supported'] is not True:
            raise WorkflowValidationError('Task not supported: ' + result['reason'])
        steps = json.loads(result['steps_json'])
        print('\n=== RAW FULL PLAN ===')
        print(json.dumps(steps, ensure_ascii=False, indent=2))
        if provider == 'local':
            steps = _normalize_local_pick_refs(steps)
        plan = validate_plan({'plan_id': uuid4().hex, 'user_text': user_text,
                              'planner': 'workflow_full_v1', 'reason': result['reason'],
                              'steps': steps})
        plan = verify_planned_detection_classes(plan, object_labels, matched_objects)
        try:
            _validate_workflow_causal_order(plan, user_text)
        except WorkflowValidationError as exc:
            if _causal_attempt != 0:
                raise
            print(f"[WORKFLOW PLAN GUARD] invalid action order: {exc}; replanning once", flush=True)
            payload["causal_replan_feedback"] = {
                "rejected_plan": plan["steps"],
                "error": str(exc),
                "instruction": (
                    "Discard the rejected steps. Generate a COMPLETE fresh plan "
                    "from S1. For single-arm drawer storage: OPEN drawer "
                    "before PICK, then PLACE and CLOSE if requested. "
                    "Re-establish a source-facing camera view before YOLO "
                    "detection, and regenerate all step IDs and $ref paths."
                ),
            }
            continue
        return plan
    raise WorkflowValidationError("Workflow causal replanning exhausted")