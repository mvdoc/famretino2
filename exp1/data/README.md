# Description of data format for Experiment 1

## `data.csv`

- `subject`: subject id
- `session`: each subject participated in two experimental sessions, at
  least 4 weeks apart
- `ntrl`: trial number; resets at the beginning of a new block, max is
  112
- `morph`: amount of morphing of the stimulus
- `pos`: angular location expressed as integer, multiple of pi/4. For
  example `2` means pi/2, or 90 degrees CCW
- `ecc`: eccentricity of the stimulus in degrees of visual angle; always
  7 in this experiment
- `rt`: reaction time
- `response`: subject response, one of {'LeftArrow', 'RightArrow'},
  corresponding to identities {'a', 'b'}
- `response_bin`: binarized response
