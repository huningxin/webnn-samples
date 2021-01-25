
'use strict';

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

/* eslint-disable */

// Noise Suppression Net 2 (NSNet2) Baseline Model for Deep Noise Suppression Challenge (DNS) 2021.
async function nsnet2(weightUrl, batchSize, frames) {
  // Constant shapes and sizes
  const HIDDEN_SIZE = 400;
  const FRAME_SIZE = 161;
  const INPUT_DIMS = [batchSize, frames, FRAME_SIZE];
  const HIDDEN_DIMS = [1, batchSize, HIDDEN_SIZE];
  const WEIGHT172_DIMS = [FRAME_SIZE, HIDDEN_SIZE];
  const WEIGHT172_SIZE = FRAME_SIZE * HIDDEN_SIZE;
  const WEIGHT_GRU_DIMS = [1, 3 * HIDDEN_SIZE, HIDDEN_SIZE];
  const WEIGHT_GRU_SIZE = 3 * HIDDEN_SIZE * HIDDEN_SIZE;
  const BIAS_GRU_DIMS = [1, 3 * HIDDEN_SIZE];
  const BIAS_GRU_SIZE = 3 * HIDDEN_SIZE;
  const FC_SIZE = 600;
  const WEIGHT215_DIMS = [HIDDEN_SIZE, FC_SIZE];
  const WEIGHT215_SIZE = HIDDEN_SIZE * FC_SIZE;
  const WEIGHT216_DIMS = [FC_SIZE, FC_SIZE];
  const WEIGHT216_SIZE = FC_SIZE * FC_SIZE;
  const WEIGHT217_DIMS = [FC_SIZE, FRAME_SIZE];
  const WEIGHT217_SIZE = FC_SIZE * FRAME_SIZE;

  // Load pre-trained constant data and initializers
  let response = await fetch(weightUrl);
  let buffer = await response.arrayBuffer();
  let byteOffset = 0;
  const weightData172 = new Float32Array(buffer, byteOffset, WEIGHT172_SIZE);
  byteOffset += WEIGHT172_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const biasDataFcIn0 = new Float32Array(buffer, byteOffset, HIDDEN_SIZE);
  byteOffset += HIDDEN_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const weightData192 = new Float32Array(buffer, byteOffset, WEIGHT_GRU_SIZE);
  byteOffset += WEIGHT_GRU_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const recurrentWeightData193 = new Float32Array(buffer, byteOffset, WEIGHT_GRU_SIZE);
  byteOffset += WEIGHT_GRU_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const biasData194 = new Float32Array(buffer, byteOffset, BIAS_GRU_SIZE);
  byteOffset += BIAS_GRU_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const recurrentBiasData194 = new Float32Array(buffer, byteOffset, BIAS_GRU_SIZE);
  byteOffset += BIAS_GRU_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const weightData212 = new Float32Array(buffer, byteOffset, WEIGHT_GRU_SIZE);
  byteOffset += WEIGHT_GRU_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const recurrentWeightData213 = new Float32Array(buffer, byteOffset, WEIGHT_GRU_SIZE);
  byteOffset += WEIGHT_GRU_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const biasData214 = new Float32Array(buffer, byteOffset, BIAS_GRU_SIZE);
  byteOffset += BIAS_GRU_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const recurrentBiasData214 = new Float32Array(buffer, byteOffset, BIAS_GRU_SIZE);
  byteOffset += BIAS_GRU_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const weightData215 = new Float32Array(buffer, byteOffset, WEIGHT215_SIZE);
  byteOffset += WEIGHT215_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const biasDataFcOut0 = new Float32Array(buffer, byteOffset, FC_SIZE);
  byteOffset += FC_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const weightData216 = new Float32Array(buffer, byteOffset, WEIGHT216_SIZE);
  byteOffset += WEIGHT216_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const biasDataFcOut2 = new Float32Array(buffer, byteOffset, FC_SIZE);
  byteOffset += FC_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const weightData217 = new Float32Array(buffer, byteOffset, WEIGHT217_SIZE);
  byteOffset += WEIGHT217_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const biasDataFcOut4 = new Float32Array(buffer, byteOffset, FRAME_SIZE);

  // Create constant operands
  const builder = navigator.ml.getNeuralNetworkContext().createModelBuilder();
  const weight172 = builder.constant({ type: 'float32', dimensions: WEIGHT172_DIMS }, weightData172);
  const biasFcIn0 = builder.constant({ type: 'float32', dimensions: [HIDDEN_SIZE] }, biasDataFcIn0);
  const weight192 = builder.constant({ type: 'float32', dimensions: WEIGHT_GRU_DIMS }, weightData192);
  const recurrentWeight193 = builder.constant({ type: 'float32', dimensions: WEIGHT_GRU_DIMS }, recurrentWeightData193);
  const bias194 = builder.constant({ type: 'float32', dimensions: BIAS_GRU_DIMS }, biasData194);
  const recurrentBias194 = builder.constant({ type: 'float32', dimensions: BIAS_GRU_DIMS }, recurrentBiasData194);
  const weight212 = builder.constant({ type: 'float32', dimensions: WEIGHT_GRU_DIMS }, weightData212);
  const recurrentWeight213 = builder.constant({ type: 'float32', dimensions: WEIGHT_GRU_DIMS }, recurrentWeightData213);
  const bias214 = builder.constant({ type: 'float32', dimensions: BIAS_GRU_DIMS }, biasData214);
  const recurrentBias214 = builder.constant({ type: 'float32', dimensions: BIAS_GRU_DIMS }, recurrentBiasData214);
  const weight215 = builder.constant({ type: 'float32', dimensions: WEIGHT215_DIMS }, weightData215);
  const biasFcOut0 = builder.constant({ type: 'float32', dimensions: [FC_SIZE] }, biasDataFcOut0);
  const weight216 = builder.constant({ type: 'float32', dimensions: WEIGHT216_DIMS }, weightData216);
  const biasFcOut2 = builder.constant({ type: 'float32', dimensions: [FC_SIZE] }, biasDataFcOut2);
  const weight217 = builder.constant({ type: 'float32', dimensions: WEIGHT217_DIMS }, weightData217);
  const biasFcOut4 = builder.constant({ type: 'float32', dimensions: [FRAME_SIZE] }, biasDataFcOut4);

  // Build up the network
  const input = builder.input('input', { type: 'float32', dimensions: INPUT_DIMS });
  const matmul18 = builder.matmul(input, weight172);
  const add19 = builder.add(matmul18, biasFcIn0);
  const relu20 = builder.relu(add19);
  const transpose31 = builder.transpose(relu20, { permutation: [1, 0, 2] });
  const initialHiddenState92 = builder.input('initialHiddenState92', { type: 'float32', dimensions: HIDDEN_DIMS });
  const [gru94, gru93] = builder.gru(transpose31, weight192, recurrentWeight193, frames, HIDDEN_SIZE,
      { bias: bias194, recurrentBias: recurrentBias194, initialHiddenState: initialHiddenState92, returnSequence: true });
  const squeeze95 = builder.squeeze(gru93, { axes: [1] });
  const initialHiddenState155 = builder.input('initialHiddenState155', { type: 'float32', dimensions: HIDDEN_DIMS });
  const [gru157, gru156] = builder.gru(squeeze95, weight212, recurrentWeight213, frames, HIDDEN_SIZE,
      { bias: bias214, recurrentBias: recurrentBias214, initialHiddenState: initialHiddenState155, returnSequence: true});
  const squeeze158 = builder.squeeze(gru156, { axes: [1] });
  const transpose159 = builder.transpose(squeeze158, { permutation: [1, 0, 2] });
  const matmul161 = builder.matmul(transpose159, weight215);
  const add162 = builder.add(matmul161, biasFcOut0);
  const relu163 = builder.relu(add162);
  const matmul165 = builder.matmul(relu163, weight216);
  const add166 = builder.add(matmul165, biasFcOut2);
  const relu167 = builder.relu(add166);
  const matmul169 = builder.matmul(relu167, weight217);
  const add170 = builder.add(matmul169, biasFcOut4);
  const output = builder.sigmoid(add170);

  // Compile the model
  const model = builder.createModel({ output, gru94, gru157 });
  return await model.compile();
}

async function run(compiledModel, inputBuffer, initialHiddenState92Buffer, initialHiddenState155Buffer) {
  // Run the compiled model with the input data
  const inputs = {
    input: { buffer: inputBuffer },
    initialHiddenState92: { buffer: initialHiddenState92Buffer },
    initialHiddenState155: { buffer: initialHiddenState155Buffer },
  };
  return await compiledModel.compute(inputs);
}

/* eslint-enable */

export class NSNet2 {
  constructor() {
    this.baseUrl_ = './';
    this.compilation_ = null;
    this.hiddenSize = 400;
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
    const response = await fetch(this.baseUrl_ + fileName);
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

  async load(url, batchSize, frames) {
    this.baseUrl_ = url;
    this.batchSize_ = batchSize;
    this.frames_ = frames;

    const nn = navigator.ml.getNeuralNetworkContext();
    const builder = nn.createModelBuilder();
    this.builder = builder;

    // Create constants
    const weight172 = await this.buildConstantByNpy('172.npy');
    const biasFcIn0 = await this.buildConstantByNpy('fc_in_0_bias.npy');
    const weight192 = await this.buildConstantByNpy('192.npy');
    const recurrentWeight193 = await this.buildConstantByNpy('193.npy');
    const data194 = await this.buildConstantByNpy('194.npy');
    const weight212 = await this.buildConstantByNpy('212.npy');
    const recurrentWeight213 = await this.buildConstantByNpy('213.npy');
    const data214 = await this.buildConstantByNpy('214.npy');
    const weight215 = await this.buildConstantByNpy('215.npy');
    const biasFcOut0 = await this.buildConstantByNpy('fc_out_0_bias.npy');
    const weight216 = await this.buildConstantByNpy('216.npy');
    const biasFcOut2 = await this.buildConstantByNpy('fc_out_2_bias.npy');
    const weight217 = await this.buildConstantByNpy('217.npy');
    const biasFcOut4 = await this.buildConstantByNpy('fc_out_4_bias.npy');

    const weightBuffer = new Float32Array(
        sizeOfShape(weight172.desc.dimensions) +
        sizeOfShape(weight192.desc.dimensions) +
        sizeOfShape(recurrentWeight193.desc.dimensions) +
        sizeOfShape(weight212.desc.dimensions) +
        sizeOfShape(recurrentWeight213.desc.dimensions) +
        sizeOfShape(weight215.desc.dimensions) +
        sizeOfShape(weight216.desc.dimensions) +
        sizeOfShape(weight217.desc.dimensions) +
        sizeOfShape(biasFcIn0.desc.dimensions) +
        sizeOfShape(data194.desc.dimensions) +
        sizeOfShape(data214.desc.dimensions) +
        sizeOfShape(biasFcOut0.desc.dimensions) +
        sizeOfShape(biasFcOut2.desc.dimensions) +
        sizeOfShape(biasFcOut4.desc.dimensions));
    let offset = 0;
    weightBuffer.set(weight172.value, offset);
    offset += sizeOfShape(weight172.desc.dimensions);
    weightBuffer.set(biasFcIn0.value, offset);
    offset += sizeOfShape(biasFcIn0.desc.dimensions);
    weightBuffer.set(weight192.value, offset);
    offset += sizeOfShape(weight192.desc.dimensions);
    weightBuffer.set(recurrentWeight193.value, offset);
    offset += sizeOfShape(recurrentWeight193.desc.dimensions);
    weightBuffer.set(data194.value, offset);
    offset += sizeOfShape(data194.desc.dimensions);
    weightBuffer.set(weight212.value, offset);
    offset += sizeOfShape(weight212.desc.dimensions);
    weightBuffer.set(recurrentWeight213.value, offset);
    offset += sizeOfShape(recurrentWeight213.desc.dimensions);
    weightBuffer.set(data214.value, offset);
    offset += sizeOfShape(data214.desc.dimensions);
    weightBuffer.set(weight215.value, offset);
    offset += sizeOfShape(weight215.desc.dimensions);
    weightBuffer.set(biasFcOut0.value, offset);
    offset += sizeOfShape(biasFcOut0.desc.dimensions);
    weightBuffer.set(weight216.value, offset);
    offset += sizeOfShape(weight216.desc.dimensions);
    weightBuffer.set(biasFcOut2.value, offset);
    offset += sizeOfShape(biasFcOut2.desc.dimensions);
    weightBuffer.set(weight217.value, offset);
    offset += sizeOfShape(weight217.desc.dimensions);
    weightBuffer.set(biasFcOut4.value, offset);
    this.weightBuffer_ = weightBuffer;
  }

  async compile(options) {
    const weightUrl = URL.createObjectURL(new Blob([this.weightBuffer_]));
    this.compilation_ = await nsnet2(weightUrl, this.batchSize_, this.frames_);
  }

  async compute(
      inputBuffer, initialHiddenState92Buffer, initialHiddenState155Buffer) {
    return await run(
        this.compilation_, inputBuffer,
        initialHiddenState92Buffer, initialHiddenState155Buffer);
  }

  dispose() {
    this.compilation_.dispose();
  }
}
