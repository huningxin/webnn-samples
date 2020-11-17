'use strict';

const nn = navigator.ml.getNeuralNetworkContext();

export class RNNoise {
  constructor(url, batchSize) {
    this.url_ = url;
    this.batchSize_ = batchSize;
    this.frames_ = 100;
    this.model_ = null;
    this.compilation_ = null;
  }

  async fetchData(fileName) {
    const response = await fetch(this.url_ + fileName);
    return new Float32Array(await response.arrayBuffer());
  }

  async load() {
    const inputDenseKernel0Data = await this.fetchData('input_dense_kernel_0.bin');
    const inputDenseBias0Data = await this.fetchData('input_dense_bias_0.bin');
    const vadGruWData = await this.fetchData('vad_gru_W.bin');
    const vadGruRData = await this.fetchData('vad_gru_R.bin');
    const vadGruBData = await this.fetchData('vad_gru_B.bin');
    const noiseGruWData = await this.fetchData('noise_gru_W.bin');
    const noiseGruRData = await this.fetchData('noise_gru_R.bin');
    const noiseGruBData = await this.fetchData('noise_gru_B.bin');
    const denoiseGruWData = await this.fetchData('denoise_gru_W.bin');
    const denoiseGruRData = await this.fetchData('denoise_gru_R.bin');
    const denoiseGruBData = await this.fetchData('denoise_gru_B.bin');
    const denoiseOutputKernel0Data = await this.fetchData('denoise_output_kernel_0.bin');
    const denoiseOutputBias0Data = await this.fetchData('denoise_output_bias_0.bin');

    const builder = nn.createModelBuilder();

    const input = builder.input('input', {type: 'float32', dimensions: [this.batchSize_, 100, 42]});
    const inputDenseKernel0 = builder.constant({type: 'float32', dimensions: [42, 24]}, inputDenseKernel0Data);
    const inputDense0 = builder.matmul(input, inputDenseKernel0);

    const inputDenseBias = builder.constant({type: 'float32', dimensions: [24]}, inputDenseBias0Data);
    const biasedTensorName2 = builder.add(inputDense0, inputDenseBias);

    const inputDenseTanh0 = builder.tanh(biasedTensorName2);

    const vadGruX = builder.transpose(inputDenseTanh0, {permutation: [1, 0, 2]});
    const vadGruW = builder.constant({type: 'float32', dimensions: [1, 72, 24]}, vadGruWData);
    const vadGruR = builder.constant({type: 'float32', dimensions: [1, 72, 24]}, vadGruRData);
    const vadGruB = builder.constant({type: 'float32', dimensions: [1, 72]}, vadGruBData.subarray(0, 72));
    const vadGruRB = builder.constant({type: 'float32', dimensions: [1, 72]}, vadGruBData.subarray(72, 144));
    const [, vadGruY] = builder.gru(vadGruX, vadGruW, vadGruR, this.frames_, 24, {bias: vadGruB, recurrentBias: vadGruRB, returnSequence: true, resetAfter: false, activations: ["sigmoid", "relu"]});

    const vadGruYTransposed = builder.transpose(vadGruY, {permutation: [2, 0, 1, 3]});

    const vadGruTranspose1 = builder.reshape(vadGruYTransposed, [-1, 100, 24]);

    const concatenate1 = builder.concat([inputDenseTanh0, vadGruTranspose1, input], 2);

    const noiseGruX = builder.transpose(concatenate1, {permutation: [1, 0, 2]});
    const noiseGruW = builder.constant({type: 'float32', dimensions: [1, 144, 90]}, noiseGruWData);
    const noiseGruR = builder.constant({type: 'float32', dimensions: [1, 144, 48]}, noiseGruRData);
    const noiseGruB = builder.constant({type: 'float32', dimensions: [1, 144]}, noiseGruBData.subarray(0, 144));
    const noiseGruRB = builder.constant({type: 'float32', dimensions: [1, 144]}, noiseGruBData.subarray(144, 288));
    const [, noiseGruY] = builder.gru(noiseGruX, noiseGruW, noiseGruR, this.frames_, 48, {bias: noiseGruB, recurrentBias: noiseGruRB, returnSequence: true, resetAfter: false, activations: ["sigmoid", "relu"]});
    const noiseGruYTransposed = builder.transpose(noiseGruY, {permutation: [2, 0, 1, 3]});

    const noiseGruTranspose1 = builder.reshape(noiseGruYTransposed, [-1, 100, 48]);

    const concatenate2 = builder.concat([vadGruTranspose1, noiseGruTranspose1, input], 2);
    
    const denoiseGruX = builder.transpose(concatenate2, {permutation: [1, 0, 2]});
    const denoiseGruW = builder.constant({type: 'float32', dimensions: [1, 288, 114]}, denoiseGruWData);
    const denoiseGruR = builder.constant({type: 'float32', dimensions: [1, 288, 96]}, denoiseGruRData);
    const denoiseGruB = builder.constant({type: 'float32', dimensions: [1, 288]}, denoiseGruBData.subarray(0, 288));
    const denoiseGruRB = builder.constant({type: 'float32', dimensions: [1, 288]}, denoiseGruBData.subarray(288, 576));
    const [, denoiseGruY] = builder.gru(denoiseGruX, denoiseGruW, denoiseGruR, this.frames_, 96, {bias: denoiseGruB, recurrentBias: denoiseGruRB, returnSequence: true, resetAfter: false, activations: ["sigmoid", "relu"]});
    const denoiseGruYTransposed = builder.transpose(denoiseGruY, {permutation: [2, 0, 1, 3]});

    const denoiseGruTranspose1 = builder.reshape(denoiseGruYTransposed, [-1, 100, 96]);

    const denoiseOutputKernel0 = builder.constant({type: 'float32', dimensions: [96, 22]}, denoiseOutputKernel0Data);
    const denoiseOutput0 = builder.matmul(denoiseGruTranspose1, denoiseOutputKernel0);

    const denoiseOutputBias0 = builder.constant({type: 'float32', dimensions: [22]}, denoiseOutputBias0Data);
    const biasedTensorName = builder.add(denoiseOutput0, denoiseOutputBias0)

    const denoiseOutput = builder.sigmoid(biasedTensorName);

    this.model_ = builder.createModel({'output': denoiseOutput});
  }

  async compile(options) {
    this.compilation_ = await this.model_.compile(options);
  }

  async compute(inputBuffer) {
    const inputs = {input: {buffer: inputBuffer}};
    const outputs = await this.compilation_.compute(inputs);
    return outputs.output;
  }

  dispose() {
    this.compilation_.dispose();
  }
}
