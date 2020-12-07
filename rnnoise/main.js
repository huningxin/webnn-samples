import {Processer} from './processer.js';
import {RNNoise} from './rnnoise.js';

let runtime = false;
let rnnoise = new RNNoise('./model/', 1);
let audio = [];

Module.onRuntimeInitialized = function() {
  runtime = true;
  console.log('WASM Runtime Ready.');
};

window.onload = async function () {
  let start = performance.now();
  await rnnoise.load();
  const loadingTime = performance.now() - start;
  console.log(`loading elapsed time: ${loadingTime.toFixed(2)} ms`);
  start = performance.now();
  await rnnoise.compile();
  const compilationTime = performance.now() - start;
  console.log(`compilation elapsed time: ${compilationTime.toFixed(2)} ms`);
  $("#denoise").attr("disabled", false);
}

async function denoise() {
  let audioContext = new AudioContext({sampleRate: 48000});
  let audioElement = document.getElementById("audio");
  let sampleRate = audioContext.sampleRate;

  if(audioContext.state != "running") {
    audioContext.resume().then(function() {
      console.log('audioContext resumed.')
    });
  }
  // audioElement.play();
  if(runtime) {
    let analyser = new Processer(audioContext, audioElement);
    let pcm = await analyser.getAudioPCMData();
    let frames = Math.ceil(pcm.length / sampleRate);
    let lastFrameSize = pcm.length - sampleRate * (frames - 1);
    // console.log("Audio PCM:", pcm);
    // let features = analyser.getRNNoiseFeatures(pcm);
    for (let i = 0; i < frames; i++) {
      let framePCM;
      if (i != (frames -1)) {
        framePCM = pcm.subarray(i * sampleRate, (i + 1) * sampleRate);
      } else {
        framePCM = new Float32Array(sampleRate).fill(0);
        for(let j = 0; j < lastFrameSize; j++) {
          framePCM[j] = pcm[i*sampleRate + j];
        }
      }
      let features = analyser.preProcessing(framePCM);
      // console.log("RNNoise features:", features);
      let inputTensor = new Float32Array(features);
      let start = performance.now();
      let outputTensor = await rnnoise.compute(inputTensor);
      const executionTime = performance.now() - start;
      // rnnoise.dispose();
      console.log(`execution elapsed time: ${executionTime.toFixed(2)} ms`);
      // console.log("RNNoise Output:", outputTensor);
      let output = analyser.postProcessing(outputTensor.buffer);
      audio.push(...output);
    }
    
    // console.log("Audio Output:", audio);
    let myArrayBuffer = audioContext.createBuffer(1, sampleRate * frames, sampleRate);
    let nowBuffering = myArrayBuffer.getChannelData(0);
    for (let i=0; i<nowBuffering.length; i++) {
      nowBuffering[i] = audio[i];
    }
    let source = audioContext.createBufferSource();
    source.buffer = myArrayBuffer;
    source.connect(audioContext.destination);
    source.start();

  } else {
    console.log('WASM Runtime ERROR!');
  }
}

function play() {
  let audioContext = new AudioContext({sampleRate: 48000});
  let myArrayBuffer = audioContext.createBuffer(1, audioContext.sampleRate * 10, audioContext.sampleRate);
  let nowBuffering = myArrayBuffer.getChannelData(0);
  for (let i=0; i<nowBuffering.length; i++) {
    nowBuffering[i] = audio[i];
  }
  let source = audioContext.createBufferSource();
  source.buffer = myArrayBuffer;
  source.connect(audioContext.destination);
  source.start();
}

let buttonElement = document.getElementById("denoise");
buttonElement.addEventListener("click", denoise, false);

let playButton = document.getElementById("play");
playButton.addEventListener("click", play, false);
