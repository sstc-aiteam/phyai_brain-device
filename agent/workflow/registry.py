"""Aggregate independent registries; action registry stays in agent.tools."""
from __future__ import annotations
from copy import deepcopy
import math


class WorkflowValidationError(ValueError):
    pass


def all_tools():
    from agent.tools import get_tool as get_action_tool
    from agent.workflow.perception_tools import get_tool as get_perception_tool
    action = get_action_tool()
    perception = get_perception_tool()
    if not isinstance(action, dict) or not isinstance(perception, dict):
        raise WorkflowValidationError("Tools Registry must be mappings")
    if any(not isinstance(k, str) or k.startswith('_') or not k.isidentifier() for k in [*action, *perception]):
        raise WorkflowValidationError("Invalid tool name")
    if set(action) & set(perception):
        raise WorkflowValidationError("Duplicate tool name in independent registries")
    result = {}
    for name, tool in [*action.items(), *perception.items()]:
        expected = 'action' if name in action else 'perception'
        if not isinstance(tool, dict) or tool.get('module') != expected:
            raise WorkflowValidationError(f"Unexpected tool module: {name}")
        specs, required = tool.get('parameters'), tool.get('required')
        if not isinstance(specs, dict) or not isinstance(required, list) or not set(required) <= set(specs):
            raise WorkflowValidationError(f"Invalid tool schema: {name}")
        result[name] = deepcopy(tool)
    return result


def prompt_tools():
    return [{"tool_name": name, **spec} for name, spec in all_tools().items()]


def check_value(key, value, spec):
    kind = spec.get('type')
    good = {
        'string': lambda v: isinstance(v, str) and bool(v.strip()),
        'number': lambda v: type(v) in (int, float) and math.isfinite(v),
        'integer': lambda v: type(v) is int,
        'boolean': lambda v: type(v) is bool,
    }
    if kind not in good or not good[kind](value):
        raise WorkflowValidationError(f"Invalid parameter type: {key} ({kind})")
    if 'enum' in spec and value not in spec['enum']:
        raise WorkflowValidationError(f"Invalid enum: {key}")
    if kind in ('number', 'integer'):
        if 'minimum' in spec and value < spec['minimum']:
            raise WorkflowValidationError(f"Below minimum: {key}")
        if 'maximum' in spec and value > spec['maximum']:
            raise WorkflowValidationError(f"Above maximum: {key}")


def is_reference(value):
    return isinstance(value, dict) and set(value) == {'$ref'} and isinstance(value['$ref'], str)


def validate_step(step, catalog, *, allow_references=False, earlier_ids=()):
    if not isinstance(step, dict) or set(step) != {'step_id', 'tool_name', 'arguments'}:
        raise WorkflowValidationError("Step must contain step_id/tool_name/arguments")
    step_id, name, args = step['step_id'], step['tool_name'], step['arguments']
    if not isinstance(step_id, str) or not step_id.isidentifier():
        raise WorkflowValidationError("Invalid step_id")
    if name not in catalog or not isinstance(args, dict):
        raise WorkflowValidationError(f"Unknown tool or invalid arguments: {name}")
    specs, required = catalog[name]['parameters'], catalog[name]['required']
    if set(args) - set(specs) or not set(required) <= set(args):
        raise WorkflowValidationError(f"Unexpected/missing parameters for {name}")
    for key, value in args.items():
        if allow_references and is_reference(value):
            ref = value['$ref']
            # Only earlier action results; no arbitrary expressions or code.
            root, separator, path = ref.partition('.')
            if not separator or not path or root not in earlier_ids:
                raise WorkflowValidationError(f"Invalid/forward ref: {ref}")
            if any(not p or p.startswith('_') or not (p.isidentifier() or p.isdecimal()) for p in path.split('.')):
                raise WorkflowValidationError(f"Invalid ref path: {ref}")
        else:
            check_value(key, value, specs[key])
    return deepcopy(step)


def validate_plan(plan, catalog=None):
    catalog = all_tools() if catalog is None else catalog
    if not isinstance(plan, dict) or not isinstance(plan.get('steps'), list) or not plan['steps']:
        raise WorkflowValidationError("Plan must contain non-empty steps")
    if len(plan['steps']) > 16:
        raise WorkflowValidationError("Plan exceeds 16 steps")
    seen = []
    checked = []
    for step in plan['steps']:
        checked_step = validate_step(step, catalog, allow_references=True, earlier_ids=seen)
        if checked_step['step_id'] in seen:
            raise WorkflowValidationError("Duplicate step_id")
        seen.append(checked_step['step_id'])
        checked.append(checked_step)
    result = dict(plan)
    result['steps'] = checked
    return result


def resolve_arguments(step, results, catalog):
    arguments = {}
    for key, value in step['arguments'].items():
        if is_reference(value):
            ref = value['$ref']
            parts = ref.split('.')
            if parts[0] not in results:
                raise WorkflowValidationError(f"Reference not available: {ref}")
            node = results[parts[0]]
            for part in parts[1:]:
                if isinstance(node, dict) and part in node:
                    node = node[part]
                elif isinstance(node, (list, tuple)) and part.isdecimal() and int(part) < len(node):
                    node = node[int(part)]
                else:
                    raise WorkflowValidationError(f"Reference missing path: {ref}")
            arguments[key] = node
        else:
            arguments[key] = value
    resolved = {'step_id': step['step_id'], 'tool_name': step['tool_name'], 'arguments': arguments}
    validate_step(resolved, catalog)
    return arguments
