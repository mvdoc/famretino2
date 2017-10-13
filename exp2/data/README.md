# Description of data format for Experiment 2

## `data.csv`

- `subject`: subject id
- `session`: each subject participated in two experimental sessions, at
  least 4 weeks apart
- `morp_type`: which of three morphs was presented to the subject; one
  of `{'ab', 'ac', 'bc'}`
- `morph`: amount of morphing of the stimulus
- `pos`: angular location expressed as integer, multiple of pi/4. For
  example `1` means pi/4, or 45 degrees CCW
- `ecc`: eccentricity of the stimulus in degrees of visual angle; always
  7 in this experiment
- `rt`: reaction time
- `response_identity`: subject's response, one of `{'a' ,'b', 'c'}`
- `third_option`: whether a subject responded with a third option when a
  particular morph was presented; for example, if a subject saw a morph
coming from `ab` and responded with `c`, then `third_option == 1`

## `questionnaire.csv`

For a more detailed description of the questionnaire please refer to the 
manuscript.

- `subject`: subject id
- `stim`: identity to which the rating refers to
- `name`: whether participants provided the correct name of the identity
- `contact_deptevent`: have you ever seen the identity at a departmental event?
- `contact_party`: have you ever seen the identity at a party?
- `contact_grouplunch`: have you ever had a group lunch with the identity?
- `contact_singlelunch`: have you ever had a one-on-one lunch with the identity?
- `contact_text`: have you ever texted the identity personally?
- `contact_email`: have you ever emailed the identity personally?
- `contact`: sum of all the contact questions above (0-6)
- `ios`: Inclusion of the Other in the Self scale (1-7)
- `wescale`: We-scale (1-7)
- `sci1`: Subjective Closeness Inventory, first question (1-7)
- `sci2`: Subjective Closeness Inventory, second question (1-7)
- `rank`: each participant ranked how familiar each identity was to them, with
  1 being most familiar, 3 being least familiar.
