# Action Completion Verification

Inspect the current RGB image and determine whether the requested action has reached its intended visible result.

Action: `{action}`
Target: `{target_hint}`
Tool description from `agent/tools.py`:
{tool_description}

## Scene-wide drawer policy (only when target_hint is absent)

This section OVERRIDES the ordinary unique-target requirement for only
`open_drawer` and `close_drawer` when Target says `not specified; scene-wide drawer policy`.
Evaluate the set of visible relevant drawers, not a single selected drawer.
Do not treat multiple drawers by itself as ambiguity. Use only current-frame evidence.

- If ANY clearly identifiable visible drawer is OPEN:
  - `open_drawer`: `completed=true`, `observed_state="target_completed"`.
  - `close_drawer`: `completed=false`, `observed_state="target_not_completed"`.
  - This observation is decisive even when other drawers are closed or unclear.
- Otherwise, if one or more drawers can be identified and ALL relevant visible
  drawer states are clearly CLOSED, with no potentially relevant ambiguous drawer:
  - `open_drawer`: `completed=false`, `observed_state="target_not_completed"`.
  - `close_drawer`: `completed=true`, `observed_state="target_completed"`.
- If no drawers are identifiable, or no open drawer is confirmed and a potentially
  relevant drawer is occluded/unclear, return `completed=null`, `observed_state="unknown"`.
- A flush, closed drawer front with no opening is evidence of CLOSED. Do not require
  motion history. Do not mistake cabinet doors, unrelated panels, or robot parts for drawers.
- When Target provides an explicit object hint, evaluate ONLY that specific target,
  using the ordinary single-target verification rules below.

## Verification

1. Identify the target using the target hint and the visible scene. Do not confuse it with similar nearby objects.
2. Determine the intended end state from the tool description. The description defines the action's intent, but is not evidence that the action succeeded.
3. Judge only what is visible in this image. Do not assume that an issued command was executed successfully.
4. Set `completed=true` only when the intended end state is clearly visible and unambiguous.
5. Set `completed=false` only when the target is identifiable and the image clearly shows that the intended end state has not been reached. Partial progress is not completion.
6. Set `completed=null` when the target or result is ambiguous, occluded, outside the image, or cannot be verified visually from a single frame. Do not turn missing evidence into `false` or `true`.
7. The target may be partially cropped near an image edge. Use any visible relevant geometry, alignment, contact, gaps, or object placement; cropping alone does not imply `unknown`.
8. A single RGB image cannot establish motion history, forces, contact stability, precise distances, or hidden states. Do not infer them without visible evidence.
9. If the tool description does not define a sufficiently clear, observable success condition, set `completed=null`.

## Output

Return only minified JSON with exactly three fields:

{"completed":true,"confidence":0.95,"observed_state":"target_completed"}

Allowed values for `observed_state`:
- `target_completed`: the specified target has visibly reached the action's intended end state.
- `target_not_completed`: the specified target is identifiable and has visibly not reached the intended end state, including partial progress.
- `unknown`: the target or completion state cannot be reliably verified from this image.

The `completed` value must match `observed_state`:
- `target_completed` requires `completed=true`.
- `target_not_completed` requires `completed=false`.
- `unknown` requires `completed=null`.

For ambiguous targets, occlusion, API uncertainty, or insufficient visual evidence, return `unknown` rather than guessing. Never generate object-specific state names such as `top_drawer_closed`.

No explanation or additional fields.
