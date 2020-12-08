import {Processer} from './processer.js';
import {RNNoise} from './rnnoise.js';

let runtime = false;
let batchSize = 1;
let rnnoise = new RNNoise('./model/', batchSize);  //Frames is fixed at 100

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
const recorderWorker = new Worker('./recorderWorker.js');

recorderWorker.postMessage({
  command: 'init',
  config: {sampleRate: 48000, numChannels: 1},
});

recorderWorker.onmessage = function( e ) {
  const blob = e.data;
  denoisedAudio.src = URL.createObjectURL(blob);
};

Module.onRuntimeInitialized = function() {
  runtime = true;
  console.log('WASM Runtime Ready.');
};

function getUrlById (audioList, id) {
  for (const audio of Object.values(audioList).flat()) {
    if (id === audio.name) {
      return audio.url;
    }
  }
  return null;
};

function log (infoElement, message, sep = false, append = true) {
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
  let audioData = [];
  let audioContext = new AudioContext({sampleRate: 48000});
  let sampleRate = audioContext.sampleRate;
  let steps = 40000;

  if(audioContext.state != "running") {
    audioContext.resume().then(function() {
      console.log('audioContext resumed.')
    });
  }
  let analyser = new Processer(audioContext, originalAudio);
  let pcm = await analyser.getAudioPCMData();
  let frames = Math.ceil(pcm.length / steps);
  let lastFrameSize = pcm.length - steps * (frames - 1);

  const processStart = performance.now();
  for (let i = 0; i < frames; i++) {
    let framePCM;
    if (i != (frames -1)) {
      framePCM = pcm.subarray(i * steps, i * steps + sampleRate);
    } else {
      framePCM = new Float32Array(sampleRate).fill(0);
      for(let j = 0; j < lastFrameSize; j++) {
        framePCM[j] = pcm[i*sampleRate + j];
      }
    }
    let start = performance.now();
    let features = analyser.preProcessing(framePCM);
    const preProcessingTime = (performance.now() - start).toFixed(2);
    let inputTensor = new Float32Array(features);
    start = performance.now();
    let outputTensor = await rnnoise.compute(inputTensor);
    const executionTime = (performance.now() - start).toFixed(2);
    // rnnoise.dispose();
    start = performance.now();
    let output = analyser.postProcessing(outputTensor.buffer);
    const postProcessingTime = (performance.now() - start).toFixed(2);
    if (i == 0 ) {
      audioData.push(...output);
    }
    else {
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
  `frames in <span class='text-primary'>${processTime}</span> ms.`, true)


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

$(".dropdown-item").click(async (e) => {
  const audioId = $(e.target).attr('id');
  if(audioId == 'browse') {
    const evt = document.createEvent('MouseEvents');
    evt.initEvent('click', true, false);
    fileInput.dispatchEvent(evt);
  } else {
    const audioUrl = getUrlById(sampleAudios, audioId);
    log(audioName, audioUrl.substring(audioUrl.lastIndexOf('/') + 1), false, false);
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
  }
  reader.readAsDataURL(event.target.files[0]);
});

window.onload = async function () {
  log(modelInfo, `Creating NSNet2 with input shape ` +
    `[${batchSize} (batch_size) x 100 (frames) x 42].`, true);
  log(modelInfo, '- Loading model...');
  let start = performance.now();
  await rnnoise.load();
  const loadingTime = (performance.now() - start).toFixed(2);
  console.log(`loading elapsed time: ${loadingTime} ms`);
  log(modelInfo, `done in <span class='text-primary'>${loadingTime}</span> ms.`, true);
  log(modelInfo, '- Compiling model...');
  start = performance.now();
  await rnnoise.compile();
  const compilationTime = (performance.now() - start).toFixed(2);
  console.log(`compilation elapsed time: ${compilationTime} ms`);
  log(modelInfo, `done in <span class='text-primary'>${compilationTime}</span> ms.`, true);
  while(1) {
    if (runtime) {
      log(modelInfo, '- DSP library Loaded.', true);
      break;
    }
  }
  log(modelInfo, 'RNNoise is <b>ready</b>.')
  $("#choose-audio").attr("disabled", false);
}
