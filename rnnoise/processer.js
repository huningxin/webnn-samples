'use strict';

export class Processer {
  constructor(audioContext, audioElement) {
    this.audioContext = audioContext;
    this.audioElement = audioElement;
  }

  async getAudioPCMData() {
    let request = new Request(this.audioElement.src);
    let response = await fetch(request);
    let audioFileData = await response.arrayBuffer();
    let audioDecodeData = await this.audioContext.decodeAudioData(audioFileData);
    let audioPCMData = audioDecodeData.getChannelData(0);
  
    return audioPCMData;
  }

  preProcessing(pcm) {
    let pcmLength = 48000;
    let featuresLength = 4200;
    let pcmPtr = Module._malloc(4 * pcmLength);
    for (let i = 0; i < pcmLength; i++) {
      Module.HEAPF32[pcmPtr / 4 + i] = pcm[i];
    }
    let getFeatures = Module.cwrap('pre_processing', 'number', ['number']);
    let featuresPtr = getFeatures(pcmPtr);
    let features = [];
  
    for (let i = 0; i < featuresLength; i++) {
      features[i] = Module.HEAPF32[(featuresPtr >> 2) + i];
    }
    Module._free(pcmPtr, featuresPtr);
  
    return features;
  }

  postProcessing(gains) {
    let audioLength = 48000;
    let gainsLength = 2200;
    let gainsPtr = Module._malloc(4 * gainsLength);
    for (let i = 0; i < gainsLength; i++) {
      Module.HEAPF32[gainsPtr / 4 + i] = gains[i];
    }
    let getAudio = Module.cwrap('post_processing', 'number', ['number']);
    let audioPtr = getAudio(gainsPtr);
    let audio = [];
  
    for (let i = 0; i < audioLength; i++) {
      audio[i] = Module.HEAPF32[(audioPtr >> 2) + i];
    }
    Module._free(gainsPtr, audioPtr);
  
    return audio;
  }
}