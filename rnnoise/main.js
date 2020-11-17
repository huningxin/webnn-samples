import {Processer} from './processer.js';
import {RNNoise} from './rnnoise.js';

let runtime = false;
let rnnoise = new RNNoise('./model/', 1);

Module.onRuntimeInitialized = function() {
  runtime = true;
  console.log('WASM Runtime Ready.');
};

window.onload = async function () {
  await rnnoise.load();
  console.log("Model loaded!");
  await rnnoise.compile();
  console.log("Model compiled!");
}

async function play() {
  let audioContext = new AudioContext();
  let audioElement = document.getElementById("audio");

  if(audioContext.state != "running") {
    audioContext.resume().then(function() {
      console.log('audioContext resumed.')
    });
  }
  audioElement.play();
  if(runtime) {
    let analyser = new Processer(audioContext, audioElement);
    let pcm = await analyser.getAudioPCMData();
    console.log("Audio PCM:", pcm);
    let features = analyser.getRNNoiseFeatures(pcm);
    console.log("RNNoise features:", features);
    
    let outputTensor = await rnnoise.compute(features);
    console.log("RNNoise Output:", outputTensor);
  } else {
    console.log('WASM Runtime ERROR!');
  }
}

let buttonElement = document.getElementById("play");
buttonElement.addEventListener("click", play, false);
