# Description of data format for Experiment 1

## `data.csv`

- `subject`: subject id
- `session`: each subject participated in two experimental sessions, at
  least 4 weeks apart
- `morp_type`: which of three morphs was presented to the subject; one
  of {'ab', 'ac', 'bc'}
- `morph`: amount of morphing of the stimulus
- `pos`: angular location expressed as integer, multiple of pi/4. For
  example `1` means pi/4, or 45 degrees CCW
- `ecc`: eccentricity of the stimulus in degrees of visual angle; always
  7 in this experiment
- `rt`: reaction time
- `response_identity`: subject's response, one of {'a' ,'b', 'c'}
- `third_option`: whether a subject responded with a third option when a
  particular morph was presented; for example, if a subject saw a morph
coming from `ab` and responded with `c`, then `third_option == 1`

## `questionnaire.csv`
