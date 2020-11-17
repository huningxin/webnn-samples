'use strict';

export class Processer {
  constructor(context, element) {
    this.audioContext = context;
    this.audioElement = element;
  }

  async getAudioPCMData() {
    let request = new Request(this.audioElement.src);
    let response = await fetch(request);
    let audioFileData = await response.arrayBuffer();
    let audioDecodeData = await this.audioContext.decodeAudioData(audioFileData);
    let audioPCMData = audioDecodeData.getChannelData(0);
  
    return audioPCMData;
  }
  
  getRNNoiseFeatures(pcm) {
    let pcmLength = 44100;
    let featuresLength = 4200;
    let pcmPtr = Module._malloc(4 * pcmLength);
    for (let i = 0; i < pcmLength; i++) {
      Module.HEAPF32[pcmPtr / 4 + i] = pcm[i];
    }
    let getFeatures = Module.cwrap('get_features', 'number', ['number']);
    let featuresPtr = getFeatures(pcmPtr);
    let features = [];
  
    for (let i = 0; i < featuresLength; i++) {
      features[i] = Module.HEAPF32[(featuresPtr >> 2) + i];
    }
    Module._free(pcmPtr, featuresPtr);
  
    return features;
  }
}