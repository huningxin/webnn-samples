'use strict';

const nn = navigator.ml.getNeuralNetworkContext();

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

export class RNNoise {
  constructor(url, batchSize) {
    this.url_ = url;
    this.batchSize_ = batchSize;
    this.frames_ = 100;
    this.model_ = null;
    this.compilation_ = null;
    this.builder = null;
  }

  async buildConstantByNpy(fileName) {
    const dataTypeMap = new Map([
      ['f2', {type: 'float16', array: Uint16Array}],
      ['f4', {type: 'float32', array: Float32Array}],
      ['f8', {type: 'float64', array: Float64Array}],
      ['i1', {type: 'int8', array: Int8Array}],
      ['i2', {type: 'int16', array: Int16Array}],
      ['i4', {type: 'int32', array: Int32Array}],
      ['i8', {type: 'int64', array: BigInt64Array}],
      ['u1', {type: 'uint8', array: Uint8Array}],
      ['u2', {type: 'uint16', array: Uint16Array}],
      ['u4', {type: 'uint32', array: Uint32Array}],
      ['u8', {type: 'uint64', array: BigUint64Array}],
    ]);
    const response = await fetch(this.url_ + fileName);
    const buffer = await response.arrayBuffer();
    const npArray = new numpy.Array(new Uint8Array(buffer));
    if (!dataTypeMap.has(npArray.dataType)) {
      throw new Error(`Data type ${npArray.dataType} is not supported.`);
    }
    const dimensions = npArray.shape;
    const type = dataTypeMap.get(npArray.dataType).type;
    const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
    const typedArray = new TypedArrayConstructor(sizeOfShape(dimensions));
    const dataView = new DataView(npArray.data.buffer);
    const littleEndian = npArray.byteOrder === '<';
    for (let i = 0; i < sizeOfShape(dimensions); ++i) {
      typedArray[i] = dataView[`get` + type[0].toUpperCase() + type.substr(1)](
          i * TypedArrayConstructor.BYTES_PER_ELEMENT, littleEndian);
    }
    return this.builder.constant({type, dimensions}, typedArray);
  }

  async load() {
    this.builder = nn.createModelBuilder();

    const inputDenseKernel0 = await this.buildConstantByNpy('input_dense_kernel_0.npy');
    const inputDenseBias0 = await this.buildConstantByNpy('input_dense_bias_0.npy');
    const vadGruW = await this.buildConstantByNpy('vad_gru_W.npy');
    const vadGruR = await this.buildConstantByNpy('vad_gru_R.npy');
    const vadGruBData = await this.buildConstantByNpy('vad_gru_B.npy');
    const noiseGruW = await this.buildConstantByNpy('noise_gru_W.npy');
    const noiseGruR = await this.buildConstantByNpy('noise_gru_R.npy');
    const noiseGruBData = await this.buildConstantByNpy('noise_gru_B.npy');
    const denoiseGruW = await this.buildConstantByNpy('denoise_gru_W.npy');
    const denoiseGruR = await this.buildConstantByNpy('denoise_gru_R.npy');
    const denoiseGruBData = await this.buildConstantByNpy('denoise_gru_B.npy');
    const denoiseOutputKernel0 = await this.buildConstantByNpy('denoise_output_kernel_0.npy');
    const denoiseOutputBias0 = await this.buildConstantByNpy('denoise_output_bias_0.npy');

    const input = this.builder.input('input', {type: 'float32', dimensions: [this.batchSize_, 100, 42]});
    const inputDense0 = this.builder.matmul(input, inputDenseKernel0);
    const biasedTensorName2 = this.builder.add(inputDense0, inputDenseBias0);
    const inputDenseTanh0 = this.builder.tanh(biasedTensorName2);
    const vadGruX = this.builder.transpose(inputDenseTanh0, {permutation: [1, 0, 2]});
    //hiddenSize = 24
    const vadGruB = this.builder.slice(vadGruBData, [0], [3 * 24], {axes: [1]});
    const vadGruRB = this.builder.slice(vadGruBData, [3 * 24], [-1], {axes: [1]});
    const vadGruInitialH = this.builder.input('vadGruInitialH', {type: 'float32', dimensions: [1, this.batchSize_, 24]});
    const [vadGruYH, vadGruY] = this.builder.gru(
      vadGruX, vadGruW, vadGruR, this.frames_, 24, {
      bias: vadGruB, recurrentBias: vadGruRB, initialHiddenState: vadGruInitialH,
      returnSequence: true, resetAfter: false, activations: ["sigmoid", "relu"]
    });
    const vadGruYTransposed = this.builder.transpose(vadGruY, {permutation: [2, 0, 1, 3]});
    const vadGruTranspose1 = this.builder.reshape(vadGruYTransposed, [-1, 100, 24]);
    const concatenate1 = this.builder.concat([inputDenseTanh0, vadGruTranspose1, input], 2);
    const noiseGruX = this.builder.transpose(concatenate1, {permutation: [1, 0, 2]});
    //hiddenSize = 48
    const noiseGruB = this.builder.slice(noiseGruBData, [0], [3 * 48], {axes: [1]});
    const noiseGruRB = this.builder.slice(noiseGruBData, [3 * 48], [-1], {axes: [1]});
    const noiseGruInitialH = this.builder.input('noiseGruInitialH', {type: 'float32', dimensions: [1, this.batchSize_, 48]});
    const [noiseGruYH, noiseGruY] = this.builder.gru(
      noiseGruX, noiseGruW, noiseGruR, this.frames_, 48, {
      bias: noiseGruB, recurrentBias: noiseGruRB, initialHiddenState: noiseGruInitialH,
      returnSequence: true, resetAfter: false, activations: ["sigmoid", "relu"]
    });
    const noiseGruYTransposed = this.builder.transpose(noiseGruY, {permutation: [2, 0, 1, 3]});
    const noiseGruTranspose1 = this.builder.reshape(noiseGruYTransposed, [-1, 100, 48]);
    const concatenate2 = this.builder.concat([vadGruTranspose1, noiseGruTranspose1, input], 2);
    const denoiseGruX = this.builder.transpose(concatenate2, {permutation: [1, 0, 2]});
    //hiddenSize = 96
    const denoiseGruB = this.builder.slice(denoiseGruBData, [0], [3 * 96], {axes: [1]});
    const denoiseGruRB = this.builder.slice(denoiseGruBData, [3 * 96], [-1], {axes: [1]});
    const denoiseGruInitialH = this.builder.input('denoiseGruInitialH', { type: 'float32', dimensions: [1, this.batchSize_, 96] });
    const [denoiseGruYH, denoiseGruY] = this.builder.gru(
      denoiseGruX, denoiseGruW, denoiseGruR, this.frames_, 96, {
      bias: denoiseGruB, recurrentBias: denoiseGruRB, initialHiddenState: denoiseGruInitialH,
      returnSequence: true, resetAfter: false, activations: ["sigmoid", "relu"]
    });
    const denoiseGruYTransposed = this.builder.transpose(denoiseGruY, {permutation: [2, 0, 1, 3]});
    const denoiseGruTranspose1 = this.builder.reshape(denoiseGruYTransposed, [-1, 100, 96]);
    const denoiseOutput0 = this.builder.matmul(denoiseGruTranspose1, denoiseOutputKernel0);
    const biasedTensorName = this.builder.add(denoiseOutput0, denoiseOutputBias0)
    const denoiseOutput = this.builder.sigmoid(biasedTensorName);

    this.model_ = this.builder.createModel({denoiseOutput, vadGruYH, noiseGruYH, denoiseGruYH});
  }

  async compile(options) {
    this.compilation_ = await this.model_.compile(options);
  }

  async compute(inputBuffer, vadGruInitialHBuffer, noiseGruInitialHBuffer, denoiseGruInitialHBuffer) {
    const inputs = {
      input: {buffer: inputBuffer},
      vadGruInitialH: {buffer: vadGruInitialHBuffer},
      noiseGruInitialH: {buffer: noiseGruInitialHBuffer},
      denoiseGruInitialH: {buffer: denoiseGruInitialHBuffer},
    };
    const outputs = await this.compilation_.compute(inputs);
    return outputs;
  }

  dispose() {
    this.compilation_.dispose();
  }
}
