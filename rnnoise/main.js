import {Processer} from './processer.js';
import {RNNoise} from './rnnoise.js';

let runtime = false;
const batchSize = 1;
const rnnoise = new RNNoise('./model/', batchSize); // Frames is fixed at 100

const sampleAudios = [{
  name: 'babbel',
  url: './audio/babbel.wav',
}, {
  name: 'car',
  url: './audio/car.wav',
}, {
  name: 'street',
  url: './audio/street.wav',
}];

const audioName = document.getElementById('audio-name');
const modelInfo = document.getElementById('info');
const DenoiseInfo = document.getElementById('denoise-info');
const fileInput = document.getElementById('file-input');
const originalAudio = document.getElementById('original-audio');
const denoisedAudio = document.getElementById('denoised-audio');
const recorderWorker = new Worker('./utils/recorderWorker.js');

recorderWorker.postMessage({
  command: 'init',
  config: {sampleRate: 48000, numChannels: 1},
});

recorderWorker.onmessage = function(e) {
  const blob = e.data;
  denoisedAudio.src = URL.createObjectURL(blob);
};

Module.onRuntimeInitialized = function() {
  runtime = true;
  console.log('WASM Runtime Ready.');
};

function getUrlById(audioList, id) {
  for (const audio of Object.values(audioList).flat()) {
    if (id === audio.name) {
      return audio.url;
    }
  }
  return null;
}

function log(infoElement, message, sep = false, append = true) {
  infoElement.innerHTML = (append ? infoElement.innerHTML : '') + message +
    (sep ? '<br>' : '');
}

originalAudio.onplay = () => {
  denoisedAudio.pause();
};

denoisedAudio.onplay = () => {
  originalAudio.pause();
};

async function denoise() {
  const audioData = [];
  const audioContext = new AudioContext({sampleRate: 48000});
  const sampleRate = audioContext.sampleRate;
  const steps = 48000;
  let vadInitialHiddenStateBuffer = new Float32Array(
      1 * batchSize * 24,
  ).fill(0);
  let noiseInitialHiddenStateBuffer = new Float32Array(
      1 * batchSize * 48,
  ).fill(0);
  let denoiseInitialHiddenStateBuffer = new Float32Array(
      1 * batchSize * 96,
  ).fill(0);

  if (audioContext.state != 'running') {
    audioContext.resume().then(function() {
      console.log('audioContext resumed.');
    });
  }
  const analyser = new Processer(audioContext, originalAudio);
  const pcm = await analyser.getAudioPCMData();
  const frames = Math.ceil(pcm.length / steps);
  const lastFrameSize = pcm.length - steps * (frames - 1);

  const processStart = performance.now();
  for (let i = 0; i < frames; i++) {
    let framePCM;
    if (i != (frames - 1)) {
      framePCM = pcm.subarray(i * steps, i * steps + sampleRate);
    } else {
      framePCM = new Float32Array(sampleRate).fill(0);
      for (let j = 0; j < lastFrameSize; j++) {
        framePCM[j] = pcm[i * sampleRate + j];
      }
    }
    let start = performance.now();
    const features = analyser.preProcessing(framePCM);
    const preProcessingTime = (performance.now() - start).toFixed(2);
    const inputBuffer = new Float32Array(features);
    start = performance.now();
    const outputs = await rnnoise.compute(
        inputBuffer, vadInitialHiddenStateBuffer,
        noiseInitialHiddenStateBuffer, denoiseInitialHiddenStateBuffer,
    );
    const executionTime = (performance.now() - start).toFixed(2);
    // rnnoise.dispose();
    vadInitialHiddenStateBuffer = outputs.vadGruYH.buffer;
    noiseInitialHiddenStateBuffer = outputs.noiseGruYH.buffer;
    denoiseInitialHiddenStateBuffer = outputs.denoiseGruYH.buffer;

    start = performance.now();
    const output = analyser.postProcessing(outputs.denoiseOutput.buffer);
    const postProcessingTime = (performance.now() - start).toFixed(2);
    if (i == 0) {
      audioData.push(...output);
    } else {
      audioData.push(...output.slice(sampleRate - steps));
    }

    log(DenoiseInfo,
        `Denoising...  ` +
      `(${Math.ceil((i + 1) / frames * 100)}%)<br>` +
      ` - preProcessing time: <span class='text-primary'>` +
      `${preProcessingTime}</span> ms.<br>` +
      ` - RNNoise compute time: <span class='text-primary'>` +
      `${executionTime}</span> ms.<br>` +
      ` - postProcessing time: <span class='text-primary'>` +
      `${postProcessingTime}</span> ms.`, true, false);
  }
  const processTime = (performance.now() - processStart).toFixed(2);
  log(DenoiseInfo, `<b>Done.</b> Processed ${frames * 100} ` +
    `frames in <span class='text-primary'>${processTime}</span> ms.`, true);


  // Send the denoised audio data for wav encoding.
  recorderWorker.postMessage({
    command: 'clear',
  });
  recorderWorker.postMessage({
    command: 'record',
    buffer: [new Float32Array(audioData)],
  });
  recorderWorker.postMessage({
    command: 'exportWAV',
    type: 'audio/wav',
  });
}

$('.dropdown-item').click(async (e) => {
  const audioId = $(e.target).attr('id');
  if (audioId == 'browse') {
    const evt = document.createEvent('MouseEvents');
    evt.initEvent('click', true, false);
    fileInput.dispatchEvent(evt);
  } else {
    const audioUrl = getUrlById(sampleAudios, audioId);
    log(audioName,
        audioUrl.substring(audioUrl.lastIndexOf('/') + 1), false, false);
    originalAudio.src = audioUrl;
    denoisedAudio.src = '';
    await denoise();
  }
});

fileInput.addEventListener('input', (event) => {
  log(audioName, event.target.files[0].name, false, false);
  const reader = new FileReader();
  reader.onload = async function(e) {
    originalAudio.src = e.target.result;
    denoisedAudio.src = '';
    await denoise();
  };
  reader.readAsDataURL(event.target.files[0]);
});

window.onload = async function() {
  log(modelInfo, `Creating RNNoise with input shape ` +
    `[${batchSize} (batch_size) x 100 (frames) x 42].`, true);
  log(modelInfo, '- Loading model...');
  let start = performance.now();
  await rnnoise.load();
  const loadingTime = (performance.now() - start).toFixed(2);
  console.log(`loading elapsed time: ${loadingTime} ms`);
  log(modelInfo,
      `done in <span class='text-primary'>${loadingTime}</span> ms.`, true);
  log(modelInfo, '- Compiling model...');
  start = performance.now();
  await rnnoise.compile();
  const compilationTime = (performance.now() - start).toFixed(2);
  console.log(`compilation elapsed time: ${compilationTime} ms`);
  log(modelInfo,
      `done in <span class='text-primary'>${compilationTime}</span> ms.`, true);
  while (1) {
    if (runtime) {
      log(modelInfo, '- DSP library Loaded.', true);
      break;
    }
  }
  log(modelInfo, 'RNNoise is <b>ready</b>.');
  $('#choose-audio').attr('disabled', false);
};
