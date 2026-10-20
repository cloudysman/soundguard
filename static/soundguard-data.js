/* SoundGuard — frontend constants. UrbanSound8K class taxonomy.
 *
 * This file holds only presentation constants. All numbers the UI shows come from the backend:
 * predictions from /api/predict and /api/predict/upload, dashboard charts from /api/dashboard.
 * There are deliberately no hard-coded predictions or metrics here — a fabricated number that
 * looks real is the failure this project is about.
 */
const SG = {};

SG.classes = [
  'air_conditioner','car_horn','children_playing','dog_bark','drilling',
  'engine_idling','gun_shot','jackhammer','siren','street_music'
];
SG.pretty = s => s.replace(/_/g,' ');

// The seven demo chips. `id` must match the backend Literal in app.py; the rest is display metadata
// plus a seed and a "character" that drive the illustrative waveform/spectrogram canvases (the
// canvases are stylised visuals, not the real signal — the prediction itself is the model's).
SG.samples = [
  { id:'dog',  name:'dog_bark.wav',        dur:4.0, seed:1337, wave:'bursts',   mel:'bursts'   },
  { id:'siren',name:'siren.wav',           dur:4.0, seed:7,    wave:'oscillate',mel:'sweep'    },
  { id:'jack', name:'jackhammer.wav',      dur:4.0, seed:99,   wave:'dense',    mel:'dense'    },
  { id:'horn', name:'car_horn.wav',        dur:4.0, seed:2024, wave:'sustained',mel:'bands'    },
  { id:'kids', name:'children_playing.wav',dur:4.0, seed:451,  wave:'busy',     mel:'scatter'  },
  { id:'gun',  name:'gun_shot.wav',        dur:4.0, seed:5150, wave:'bursts',   mel:'bursts'   },
  { id:'ac',   name:'air_conditioner.wav', dur:4.0, seed:60,   wave:'sustained',mel:'bands'    },
];

SG.statusMeta = {
  reliable:  { name:'Reliable',  icon:'✓' },
  uncertain: { name:'Uncertain', icon:'!' },
  reject:    { name:'Reject',    icon:'✕' },
};

// The colour of each quality meter must agree with the decision the backend actually took, so
// these are a copy of the thresholds in rejection.py and have to be changed with it. They are the
// recalibrated set the demonstration uses, not the audited set of Table 3.2 in the report: a meter
// keyed to the audited numbers turned red on recordings the shipped policy calls reliable.
// tests_fe/test_l4_edge.py asserts that this copy still matches rejection.py.
SG.gateThresholds = {
  snr:  { bad: 2.0,   warn: 4.0   },   // decibels, lower is worse
  lvl:  { bad: -50.0, warn: -40.0 },   // decibels relative to full scale, lower is worse
  clip: { bad: 30.0,  warn: 12.0  },   // percent of samples near full scale, higher is worse
};

// The short tag names the cause the backend reported, not the state. The two are not the same:
// the recording axis can put a clip into the uncertain state while the model is at full
// confidence, and a tag fixed to the state then read "high uncertainty" beside a confidence of
// 100%. The state is already carried by the name and the colour of the card.
SG.causeTag = {
  noise:  'noise floor too close',
  quiet:  'level too low',
  clip:   'samples clipped',
  uncert: 'model hesitating',
  clean:  'input accepted',
};

// Turn a backend result (status + dominant cause) into a human-readable reason + suggested action.
SG.reasonText = function(r){
  const lead = SG.pretty(SG.classes[r.order[0][0]]);
  if (r.status === 'reliable')
    return { reason:`Clean input — the model confidently predicts “${lead}”.`,
             action:`Result is safe to use. Confidence is well inside the reliable zone.` };
  const map = {
    noise:  { reason:`Too noisy — estimated SNR is only ${r.snrDb.toFixed(0)} dB, swamping the signal.`,
              action:`Record again closer to the source, or reduce background noise.` },
    quiet:  { reason:`Signal too quiet — peak level ${r.levelDb.toFixed(0)} dBFS is near the noise floor.`,
              action:`Move the microphone closer to the source or raise the input gain.` },
    clip:   { reason:`Clipping distortion — ${r.clipPct.toFixed(0)}% of samples are saturated.`,
              action:`Lower the input level to avoid clipping, then capture again.` },
    uncert: { reason:`Model is hesitating between classes (max-softmax ${(r.conf*100).toFixed(0)}%).`,
              action:`Capture a longer, cleaner clip so the prediction can settle.` },
    clean:  { reason:`Acceptable input but elevated uncertainty.`,
              action:`Treat the label as a hint, not a decision.` },
  };
  return map[r.cause] || map.clean;
};
