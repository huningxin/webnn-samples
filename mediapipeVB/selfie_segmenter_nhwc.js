export class SelfieSegmenterNhwc {

  constructor() {

    this.graph_ = null;

    this.context_ = null;

    this.inputTensors_ = {};

    this.outputTensors_ = {};

  }



  async build(contextOptions) {

    // Load weights ArrayBuffer from selfie_segmenter_nhwc.bin
    async function loadWeightsArrayBuffer() {
        const response = await fetch('selfie_segmenter_nhwc.bin');
        if (!response.ok) {
            throw new Error('Failed to fetch weights: ' + response.statusText);
        }
        return await response.arrayBuffer();
    }

    const weights_array_buffer = await loadWeightsArrayBuffer();

    this.context_ = await navigator.ml.createContext(contextOptions);
    const builder = new MLGraphBuilder(this.context_);


    // Create graph constant operands.

    const scales__125 = builder.constant(
        {dataType: 'float32', shape: [4]},
        new Float32Array([1.0, 1.0, 2.0, 2.0])
    );

    const new_shape__369 = builder.constant(
        {dataType: 'int64', shape: [4]},
        new BigInt64Array(weights_array_buffer, 0, 32 / BigInt64Array.BYTES_PER_ELEMENT)
    );

    const var_mul_3_y_0 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array(weights_array_buffer, 32, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_9_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 36, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_8_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 100, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_7_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 196, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_6_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 580, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_5_y_0 = builder.constant(
        {dataType: 'float32', shape: [128]},
        new Float32Array(weights_array_buffer, 964, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_4_y_0 = builder.constant(
        {dataType: 'float32', shape: [128]},
        new Float32Array(weights_array_buffer, 1476, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_3_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 1988, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_2_y_0 = builder.constant(
        {dataType: 'float32', shape: [88]},
        new Float32Array(weights_array_buffer, 2372, 352 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_10_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 2724, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [72]},
        new Float32Array(weights_array_buffer, 2788, 288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_depthwise_conv2d_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 3076, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_transpose_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 1, 2, 2]},
        new Float32Array(weights_array_buffer, 3140, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_9_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 3396, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_8_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 3780, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_7_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [88]},
        new Float32Array(weights_array_buffer, 3876, 352 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_6_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 4228, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_5_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [72]},
        new Float32Array(weights_array_buffer, 4324, 288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_4_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 4612, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_42_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 4676, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_41_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 4740, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_40_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 4804, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_3_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 4868, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_39_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 4932, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_38_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 4996, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_37_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 5060, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_36_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 5124, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_35_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 5188, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_34_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 5252, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_33_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 5348, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_32_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 5444, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_31_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 5540, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_30_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [128]},
        new Float32Array(weights_array_buffer, 5636, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_2_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [8]},
        new Float32Array(weights_array_buffer, 6148, 32 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_29_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [128]},
        new Float32Array(weights_array_buffer, 6180, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_28_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [32]},
        new Float32Array(weights_array_buffer, 6692, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_27_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 6820, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_26_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 7204, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_25_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 7300, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_24_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [32]},
        new Float32Array(weights_array_buffer, 7684, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_23_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 7812, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_22_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 8196, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_21_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 8292, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_20_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [32]},
        new Float32Array(weights_array_buffer, 8676, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_1_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 8804, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_19_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [128]},
        new Float32Array(weights_array_buffer, 8868, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_18_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [32]},
        new Float32Array(weights_array_buffer, 9380, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_17_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [128]},
        new Float32Array(weights_array_buffer, 9508, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_16_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [32]},
        new Float32Array(weights_array_buffer, 10020, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_15_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [128]},
        new Float32Array(weights_array_buffer, 10148, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_14_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [32]},
        new Float32Array(weights_array_buffer, 10660, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_13_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [128]},
        new Float32Array(weights_array_buffer, 10788, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_12_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [32]},
        new Float32Array(weights_array_buffer, 11300, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_11_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [96]},
        new Float32Array(weights_array_buffer, 11428, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_10_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [24]},
        new Float32Array(weights_array_buffer, 11812, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_1_y_0 = builder.constant(
        {dataType: 'float32', shape: [16]},
        new Float32Array(weights_array_buffer, 11908, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__408 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array(weights_array_buffer, 11972, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__407 = builder.constant(
        {dataType: 'float32', shape: [96, 1, 5, 5]},
        new Float32Array(weights_array_buffer, 11976, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__402 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array(weights_array_buffer, 21576, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__395 = builder.constant(
        {dataType: 'float32', shape: [96, 1, 5, 5]},
        new Float32Array(weights_array_buffer, 21580, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__394 = builder.constant(
        {dataType: 'float32', shape: [24, 1, 3, 3]},
        new Float32Array(weights_array_buffer, 31180, 864 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__391 = builder.constant(
        {dataType: 'float32', shape: [16, 1, 3, 3]},
        new Float32Array(weights_array_buffer, 32044, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__389 = builder.constant(
        {dataType: 'float32', shape: [16, 1, 3, 3]},
        new Float32Array(weights_array_buffer, 32620, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__387 = builder.constant(
        {dataType: 'float32', shape: [72, 1, 3, 3]},
        new Float32Array(weights_array_buffer, 33196, 2592 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__385 = builder.constant(
        {dataType: 'float32', shape: [88, 1, 3, 3]},
        new Float32Array(weights_array_buffer, 35788, 3168 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__383 = builder.constant(
        {dataType: 'float32', shape: [96, 1, 5, 5]},
        new Float32Array(weights_array_buffer, 38956, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__381 = builder.constant(
        {dataType: 'float32', shape: [128, 1, 5, 5]},
        new Float32Array(weights_array_buffer, 48556, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__379 = builder.constant(
        {dataType: 'float32', shape: [128, 1, 5, 5]},
        new Float32Array(weights_array_buffer, 61356, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );

    const const_fold_opt__377 = builder.constant(
        {dataType: 'float32', shape: [16, 1, 3, 3]},
        new Float32Array(weights_array_buffer, 74156, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_9_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 88, 1, 1]},
        new Float32Array(weights_array_buffer, 74732, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_8_filter_0 = builder.constant(
        {dataType: 'float32', shape: [88, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 83180, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_7_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 72, 1, 1]},
        new Float32Array(weights_array_buffer, 91628, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_6_filter_0 = builder.constant(
        {dataType: 'float32', shape: [72, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 98540, 4608 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_5_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 103148, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_43_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 104172, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_42_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 105196, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_41_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 106220, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_40_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 107244, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_4_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 8, 1, 1]},
        new Float32Array(weights_array_buffer, 108268, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_39_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 108780, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_38_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 109804, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_37_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 110828, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_36_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 111852, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_35_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 113388, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_34_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 115692, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_33_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 117996, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_32_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 128, 1, 1]},
        new Float32Array(weights_array_buffer, 120300, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_31_filter_0 = builder.constant(
        {dataType: 'float32', shape: [128, 32, 1, 1]},
        new Float32Array(weights_array_buffer, 132588, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_30_filter_0 = builder.constant(
        {dataType: 'float32', shape: [128, 32, 1, 1]},
        new Float32Array(weights_array_buffer, 148972, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_3_filter_0 = builder.constant(
        {dataType: 'float32', shape: [8, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 165356, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_29_filter_0 = builder.constant(
        {dataType: 'float32', shape: [32, 96, 1, 1]},
        new Float32Array(weights_array_buffer, 165868, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_28_filter_0 = builder.constant(
        {dataType: 'float32', shape: [96, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 178156, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_27_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 96, 1, 1]},
        new Float32Array(weights_array_buffer, 187372, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_26_filter_0 = builder.constant(
        {dataType: 'float32', shape: [96, 32, 1, 1]},
        new Float32Array(weights_array_buffer, 196588, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_25_filter_0 = builder.constant(
        {dataType: 'float32', shape: [32, 96, 1, 1]},
        new Float32Array(weights_array_buffer, 208876, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_24_filter_0 = builder.constant(
        {dataType: 'float32', shape: [96, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 221164, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_23_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 96, 1, 1]},
        new Float32Array(weights_array_buffer, 230380, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_22_filter_0 = builder.constant(
        {dataType: 'float32', shape: [96, 32, 1, 1]},
        new Float32Array(weights_array_buffer, 239596, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_21_filter_0 = builder.constant(
        {dataType: 'float32', shape: [32, 128, 1, 1]},
        new Float32Array(weights_array_buffer, 251884, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_20_filter_0 = builder.constant(
        {dataType: 'float32', shape: [128, 32, 1, 1]},
        new Float32Array(weights_array_buffer, 268268, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_2_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 16, 1, 1]},
        new Float32Array(weights_array_buffer, 284652, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_19_filter_0 = builder.constant(
        {dataType: 'float32', shape: [32, 128, 1, 1]},
        new Float32Array(weights_array_buffer, 285676, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_18_filter_0 = builder.constant(
        {dataType: 'float32', shape: [128, 32, 1, 1]},
        new Float32Array(weights_array_buffer, 302060, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_17_filter_0 = builder.constant(
        {dataType: 'float32', shape: [32, 128, 1, 1]},
        new Float32Array(weights_array_buffer, 318444, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_16_filter_0 = builder.constant(
        {dataType: 'float32', shape: [128, 32, 1, 1]},
        new Float32Array(weights_array_buffer, 334828, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_15_filter_0 = builder.constant(
        {dataType: 'float32', shape: [32, 128, 1, 1]},
        new Float32Array(weights_array_buffer, 351212, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_14_filter_0 = builder.constant(
        {dataType: 'float32', shape: [128, 32, 1, 1]},
        new Float32Array(weights_array_buffer, 367596, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_13_filter_0 = builder.constant(
        {dataType: 'float32', shape: [32, 96, 1, 1]},
        new Float32Array(weights_array_buffer, 383980, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_12_filter_0 = builder.constant(
        {dataType: 'float32', shape: [96, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 396268, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_11_filter_0 = builder.constant(
        {dataType: 'float32', shape: [24, 96, 1, 1]},
        new Float32Array(weights_array_buffer, 405484, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_10_filter_0 = builder.constant(
        {dataType: 'float32', shape: [96, 24, 1, 1]},
        new Float32Array(weights_array_buffer, 414700, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv2D_filter_0 = builder.constant(
        {dataType: 'float32', shape: [16, 3, 3, 3]},
        new Float32Array(weights_array_buffer, 423916, 1728 / Float32Array.BYTES_PER_ELEMENT)
    );

    // Create graph input operands and tensors.

    // Transpose input from NCHW -> NHWC.

    const input = builder.transpose(
        builder.input('input', {dataType: 'float32', shape: [1, 144, 256, 3]}),
        { permutation: [0, 2, 3, 1] }
    );

    this.inputTensors_['input'] = await this.context_.createTensor(
        {dataType: 'float32', shape: [1, 144, 256, 3], writable: true}
    );

    // Create graph operators.

    const var_Conv2D__6_0 = builder.transpose(
        input,
        { permutation: [0, 3, 1, 2] }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 3, 3, 3]},
        new Float32Array(weights_array_buffer, 423916, 1728 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__158_0 = builder.conv2d(
        var_Conv2D__6_0, var_Conv2D_filter_0_transposed,
        {
            bias: var_conv2d_1_y_0, strides: [2, 2], padding: [0, 1, 0, 1], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_0 = builder.add(var_Conv__158_0, const_fold_opt__408);

    const var_292 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_293 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_0 = builder.clamp(
        var_add_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_0 = builder.mul(var_mul_3_y_0, var_Relu6_0);

    const var_mul_1_0 = builder.mul(var_Conv__158_0, var_mul_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_2_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 284652, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__161_0 = builder.conv2d(
        var_mul_1_0, var_Conv2D_2_filter_0_transposed,
        {
            bias: var_conv2d_1_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_0 = builder.relu(var_Conv__161_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__377_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 3, 3, 16]},
        new Float32Array(weights_array_buffer, 74156, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__162_0 = builder.conv2d(
        var_re_lu_0, const_fold_opt__377_transposed,
        {
            bias: var_depthwise_conv2d_y_0, strides: [2, 2], padding: [0, 1, 0, 1], dilations: [1, 1], groups: 16, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_1_0 = builder.relu(var_Conv__162_0);

    const var_global_average_pooling2d_0 = builder.averagePool2d(
        var_re_lu_1_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_3_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [8, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 165356, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__165_0 = builder.conv2d(
        var_global_average_pooling2d_0, var_Conv2D_3_filter_0_transposed,
        {
            bias: var_conv2d_2_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_2_0 = builder.relu(var_Conv__165_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_4_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 8]},
        new Float32Array(weights_array_buffer, 108268, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__166_0 = builder.conv2d(
        var_re_lu_2_0, var_Conv2D_4_filter_0_transposed,
        {
            bias: var_conv2d_3_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_0 = builder.sigmoid(var_Conv__166_0);

    const var_multiply_0 = builder.mul(var_re_lu_1_0, var_activation_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_5_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 103148, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__167_0 = builder.conv2d(
        var_multiply_0, var_Conv2D_5_filter_0_transposed,
        {
            bias: var_conv2d_4_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_6_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [72, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 98540, 4608 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__171_0 = builder.conv2d(
        var_Conv__167_0, var_Conv2D_6_filter_0_transposed,
        {
            bias: var_conv2d_5_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_3_0 = builder.relu(var_Conv__171_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__387_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 3, 3, 72]},
        new Float32Array(weights_array_buffer, 33196, 2592 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__172_0 = builder.conv2d(
        var_re_lu_3_0, const_fold_opt__387_transposed,
        {
            bias: var_depthwise_conv2d_1_y_0, strides: [2, 2], padding: [0, 1, 0, 1], dilations: [1, 1], groups: 72, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_4_0 = builder.relu(var_Conv__172_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_7_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 72]},
        new Float32Array(weights_array_buffer, 91628, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__173_0 = builder.conv2d(
        var_re_lu_4_0, var_Conv2D_7_filter_0_transposed,
        {
            bias: var_conv2d_6_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_8_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [88, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 83180, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__176_0 = builder.conv2d(
        var_Conv__173_0, var_Conv2D_8_filter_0_transposed,
        {
            bias: var_conv2d_7_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_5_0 = builder.relu(var_Conv__176_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__385_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 3, 3, 88]},
        new Float32Array(weights_array_buffer, 35788, 3168 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__177_0 = builder.conv2d(
        var_re_lu_5_0, const_fold_opt__385_transposed,
        {
            bias: var_depthwise_conv2d_2_y_0, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 88, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_6_0 = builder.relu(var_Conv__177_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_9_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 88]},
        new Float32Array(weights_array_buffer, 74732, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__178_0 = builder.conv2d(
        var_re_lu_6_0, var_Conv2D_9_filter_0_transposed,
        {
            bias: var_conv2d_8_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add__xeno_compat__1_0 = builder.add(var_Conv__178_0, var_Conv__173_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_10_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [96, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 414700, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__183_0 = builder.conv2d(
        var_add__xeno_compat__1_0, var_Conv2D_10_filter_0_transposed,
        {
            bias: var_conv2d_9_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_1_0 = builder.add(var_Conv__183_0, const_fold_opt__408);

    const var_294 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_295 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_1_0 = builder.clamp(
        var_add_1_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_2_0 = builder.mul(var_mul_3_y_0, var_Relu6_1_0);

    const var_mul_3_0 = builder.mul(var_Conv__183_0, var_mul_2_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__383_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 5, 5, 96]},
        new Float32Array(weights_array_buffer, 38956, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__186_0 = builder.conv2d(
        var_mul_3_0, const_fold_opt__383_transposed,
        {
            bias: var_depthwise_conv2d_3_y_0, strides: [2, 2], padding: [1, 2, 1, 2], dilations: [1, 1], groups: 96, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_add_2_0 = builder.add(var_Conv__186_0, const_fold_opt__408);

    const var_296 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_297 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_2_0 = builder.clamp(
        var_add_2_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_4_0 = builder.mul(var_mul_3_y_0, var_Relu6_2_0);

    const var_mul_5_0 = builder.mul(var_Conv__186_0, var_mul_4_0);

    const var_global_average_pooling2d_1_0 = builder.averagePool2d(
        var_mul_5_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_11_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 96]},
        new Float32Array(weights_array_buffer, 405484, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__189_0 = builder.conv2d(
        var_global_average_pooling2d_1_0, var_Conv2D_11_filter_0_transposed,
        {
            bias: var_conv2d_10_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_7_0 = builder.relu(var_Conv__189_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_12_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [96, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 396268, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__190_0 = builder.conv2d(
        var_re_lu_7_0, var_Conv2D_12_filter_0_transposed,
        {
            bias: var_conv2d_11_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_1_0 = builder.sigmoid(var_Conv__190_0);

    const var_multiply_1_0 = builder.mul(var_mul_5_0, var_activation_1_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_13_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [32, 1, 1, 96]},
        new Float32Array(weights_array_buffer, 383980, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__191_0 = builder.conv2d(
        var_multiply_1_0, var_Conv2D_13_filter_0_transposed,
        {
            bias: var_conv2d_12_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_14_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [128, 1, 1, 32]},
        new Float32Array(weights_array_buffer, 367596, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__194_0 = builder.conv2d(
        var_Conv__191_0, var_Conv2D_14_filter_0_transposed,
        {
            bias: var_conv2d_13_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_3_0 = builder.add(var_Conv__194_0, const_fold_opt__408);

    const var_298 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_299 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_3_0 = builder.clamp(
        var_add_3_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_6_0 = builder.mul(var_mul_3_y_0, var_Relu6_3_0);

    const var_mul_7_0 = builder.mul(var_Conv__194_0, var_mul_6_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__381_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 5, 5, 128]},
        new Float32Array(weights_array_buffer, 48556, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__197_0 = builder.conv2d(
        var_mul_7_0, const_fold_opt__381_transposed,
        {
            bias: var_depthwise_conv2d_4_y_0, strides: [1, 1], padding: [2, 2, 2, 2], dilations: [1, 1], groups: 128, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_add_4_0 = builder.add(var_Conv__197_0, const_fold_opt__408);

    const var_300 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_301 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_4_0 = builder.clamp(
        var_add_4_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_8_0 = builder.mul(var_mul_3_y_0, var_Relu6_4_0);

    const var_mul_9_0 = builder.mul(var_Conv__197_0, var_mul_8_0);

    const var_global_average_pooling2d_2_0 = builder.averagePool2d(
        var_mul_9_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_15_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [32, 1, 1, 128]},
        new Float32Array(weights_array_buffer, 351212, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__200_0 = builder.conv2d(
        var_global_average_pooling2d_2_0, var_Conv2D_15_filter_0_transposed,
        {
            bias: var_conv2d_14_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_8_0 = builder.relu(var_Conv__200_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_16_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [128, 1, 1, 32]},
        new Float32Array(weights_array_buffer, 334828, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__201_0 = builder.conv2d(
        var_re_lu_8_0, var_Conv2D_16_filter_0_transposed,
        {
            bias: var_conv2d_15_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_2_0 = builder.sigmoid(var_Conv__201_0);

    const var_multiply_2_0 = builder.mul(var_mul_9_0, var_activation_2_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_17_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [32, 1, 1, 128]},
        new Float32Array(weights_array_buffer, 318444, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__202_0 = builder.conv2d(
        var_multiply_2_0, var_Conv2D_17_filter_0_transposed,
        {
            bias: var_conv2d_16_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_1__xeno_compat__1_0 = builder.add(var_Conv__202_0, var_Conv__191_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_18_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [128, 1, 1, 32]},
        new Float32Array(weights_array_buffer, 302060, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__205_0 = builder.conv2d(
        var_add_1__xeno_compat__1_0, var_Conv2D_18_filter_0_transposed,
        {
            bias: var_conv2d_17_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_5_0 = builder.add(var_Conv__205_0, const_fold_opt__408);

    const var_302 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_303 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_5_0 = builder.clamp(
        var_add_5_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_10_0 = builder.mul(var_mul_3_y_0, var_Relu6_5_0);

    const var_mul_11_0 = builder.mul(var_Conv__205_0, var_mul_10_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__379_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 5, 5, 128]},
        new Float32Array(weights_array_buffer, 61356, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__208_0 = builder.conv2d(
        var_mul_11_0, const_fold_opt__379_transposed,
        {
            bias: var_depthwise_conv2d_5_y_0, strides: [1, 1], padding: [2, 2, 2, 2], dilations: [1, 1], groups: 128, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_add_6_0 = builder.add(var_Conv__208_0, const_fold_opt__408);

    const var_304 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_305 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_6_0 = builder.clamp(
        var_add_6_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_12_0 = builder.mul(var_mul_3_y_0, var_Relu6_6_0);

    const var_mul_13_0 = builder.mul(var_Conv__208_0, var_mul_12_0);

    const var_global_average_pooling2d_3_0 = builder.averagePool2d(
        var_mul_13_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_19_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [32, 1, 1, 128]},
        new Float32Array(weights_array_buffer, 285676, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__211_0 = builder.conv2d(
        var_global_average_pooling2d_3_0, var_Conv2D_19_filter_0_transposed,
        {
            bias: var_conv2d_18_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_9_0 = builder.relu(var_Conv__211_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_20_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [128, 1, 1, 32]},
        new Float32Array(weights_array_buffer, 268268, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__212_0 = builder.conv2d(
        var_re_lu_9_0, var_Conv2D_20_filter_0_transposed,
        {
            bias: var_conv2d_19_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_3_0 = builder.sigmoid(var_Conv__212_0);

    const var_multiply_3_0 = builder.mul(var_mul_13_0, var_activation_3_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_21_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [32, 1, 1, 128]},
        new Float32Array(weights_array_buffer, 251884, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__213_0 = builder.conv2d(
        var_multiply_3_0, var_Conv2D_21_filter_0_transposed,
        {
            bias: var_conv2d_20_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_2__xeno_compat__1_0 = builder.add(var_Conv__213_0, var_add_1__xeno_compat__1_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_22_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [96, 1, 1, 32]},
        new Float32Array(weights_array_buffer, 239596, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__216_0 = builder.conv2d(
        var_add_2__xeno_compat__1_0, var_Conv2D_22_filter_0_transposed,
        {
            bias: var_conv2d_21_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_7_0 = builder.add(var_Conv__216_0, const_fold_opt__408);

    const var_306 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_307 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_7_0 = builder.clamp(
        var_add_7_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_14_0 = builder.mul(var_mul_3_y_0, var_Relu6_7_0);

    const var_mul_15_0 = builder.mul(var_Conv__216_0, var_mul_14_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__395_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 5, 5, 96]},
        new Float32Array(weights_array_buffer, 21580, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__219_0 = builder.conv2d(
        var_mul_15_0, const_fold_opt__395_transposed,
        {
            bias: var_depthwise_conv2d_6_y_0, strides: [1, 1], padding: [2, 2, 2, 2], dilations: [1, 1], groups: 96, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_add_8_0 = builder.add(var_Conv__219_0, const_fold_opt__408);

    const var_308 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_309 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_8_0 = builder.clamp(
        var_add_8_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_16_0 = builder.mul(var_mul_3_y_0, var_Relu6_8_0);

    const var_mul_17_0 = builder.mul(var_Conv__219_0, var_mul_16_0);

    const var_global_average_pooling2d_4_0 = builder.averagePool2d(
        var_mul_17_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_23_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 96]},
        new Float32Array(weights_array_buffer, 230380, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__222_0 = builder.conv2d(
        var_global_average_pooling2d_4_0, var_Conv2D_23_filter_0_transposed,
        {
            bias: var_conv2d_22_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_10_0 = builder.relu(var_Conv__222_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_24_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [96, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 221164, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__223_0 = builder.conv2d(
        var_re_lu_10_0, var_Conv2D_24_filter_0_transposed,
        {
            bias: var_conv2d_23_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_4_0 = builder.sigmoid(var_Conv__223_0);

    const var_multiply_4_0 = builder.mul(var_mul_17_0, var_activation_4_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_25_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [32, 1, 1, 96]},
        new Float32Array(weights_array_buffer, 208876, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__224_0 = builder.conv2d(
        var_multiply_4_0, var_Conv2D_25_filter_0_transposed,
        {
            bias: var_conv2d_24_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_3__xeno_compat__1_0 = builder.add(var_Conv__224_0, var_add_2__xeno_compat__1_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_26_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [96, 1, 1, 32]},
        new Float32Array(weights_array_buffer, 196588, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__227_0 = builder.conv2d(
        var_add_3__xeno_compat__1_0, var_Conv2D_26_filter_0_transposed,
        {
            bias: var_conv2d_25_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_9_0 = builder.add(var_Conv__227_0, const_fold_opt__408);

    const var_310 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_311 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_9_0 = builder.clamp(
        var_add_9_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_18_0 = builder.mul(var_mul_3_y_0, var_Relu6_9_0);

    const var_mul_19_0 = builder.mul(var_Conv__227_0, var_mul_18_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__407_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 5, 5, 96]},
        new Float32Array(weights_array_buffer, 11976, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__230_0 = builder.conv2d(
        var_mul_19_0, const_fold_opt__407_transposed,
        {
            bias: var_depthwise_conv2d_7_y_0, strides: [1, 1], padding: [2, 2, 2, 2], dilations: [1, 1], groups: 96, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_add_10_0 = builder.add(var_Conv__230_0, const_fold_opt__408);

    const var_312 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0])
    );

    const var_313 = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([6.0])
    );

    const var_Relu6_10_0 = builder.clamp(
        var_add_10_0,
        {
            minValue: 0.0,
            maxValue: 6.0
        }
    );

    const var_mul_20_0 = builder.mul(var_mul_3_y_0, var_Relu6_10_0);

    const var_mul_21_0 = builder.mul(var_Conv__230_0, var_mul_20_0);

    const var_global_average_pooling2d_5_0 = builder.averagePool2d(
        var_mul_21_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_27_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 96]},
        new Float32Array(weights_array_buffer, 187372, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__233_0 = builder.conv2d(
        var_global_average_pooling2d_5_0, var_Conv2D_27_filter_0_transposed,
        {
            bias: var_conv2d_26_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_11_0 = builder.relu(var_Conv__233_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_28_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [96, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 178156, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__234_0 = builder.conv2d(
        var_re_lu_11_0, var_Conv2D_28_filter_0_transposed,
        {
            bias: var_conv2d_27_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_5_0 = builder.sigmoid(var_Conv__234_0);

    const var_multiply_5_0 = builder.mul(var_mul_21_0, var_activation_5_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_29_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [32, 1, 1, 96]},
        new Float32Array(weights_array_buffer, 165868, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__235_0 = builder.conv2d(
        var_multiply_5_0, var_Conv2D_29_filter_0_transposed,
        {
            bias: var_conv2d_28_1_y_0, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_4__xeno_compat__1_0 = builder.add(var_Conv__235_0, var_add_3__xeno_compat__1_0);

    const var_global_average_pooling2d_6_0 = builder.averagePool2d(
        var_add_4__xeno_compat__1_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_31_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [128, 1, 1, 32]},
        new Float32Array(weights_array_buffer, 132588, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__238_0 = builder.conv2d(
        var_global_average_pooling2d_6_0, var_Conv2D_31_filter_0_transposed,
        {
            bias: var_conv2d_30_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_6_0 = builder.sigmoid(var_Conv__238_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_30_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [128, 1, 1, 32]},
        new Float32Array(weights_array_buffer, 148972, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__239_0 = builder.conv2d(
        var_add_4__xeno_compat__1_0, var_Conv2D_30_filter_0_transposed,
        {
            bias: var_conv2d_29_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_12_0 = builder.relu(var_Conv__239_0);

    const var_multiply_6_0 = builder.mul(var_re_lu_12_0, var_activation_6_0);

    const var_314 = builder.constant(
        {dataType: 'float32', shape: [8]},
        new Float32Array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    );

    const var_289 = builder.resample2d(
        var_multiply_6_0,
        {
            mode: 'linear', scales: [2.0, 2.0], sizes: undefined, axes: [1, 2]
        }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_32_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 128]},
        new Float32Array(weights_array_buffer, 120300, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__240_0 = builder.conv2d(
        var_289, var_Conv2D_32_filter_0_transposed,
        {
            bias: var_conv2d_31_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_5__xeno_compat__1_0 = builder.add(var_add__xeno_compat__1_0, var_Conv__240_0);

    const var_global_average_pooling2d_7_0 = builder.averagePool2d(
        var_add_5__xeno_compat__1_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_33_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 117996, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__243_0 = builder.conv2d(
        var_global_average_pooling2d_7_0, var_Conv2D_33_filter_0_transposed,
        {
            bias: var_conv2d_32_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_13_0 = builder.relu(var_Conv__243_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_34_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 115692, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__244_0 = builder.conv2d(
        var_re_lu_13_0, var_Conv2D_34_filter_0_transposed,
        {
            bias: var_conv2d_33_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_7_0 = builder.sigmoid(var_Conv__244_0);

    const var_multiply_7_0 = builder.mul(var_add__xeno_compat__1_0, var_activation_7_0);

    const var_add_6__xeno_compat__1_0 = builder.add(var_multiply_7_0, var_Conv__240_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_35_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [24, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 113388, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__245_0 = builder.conv2d(
        var_add_6__xeno_compat__1_0, var_Conv2D_35_filter_0_transposed,
        {
            bias: var_conv2d_34_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_14_0 = builder.relu(var_Conv__245_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__394_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 3, 3, 24]},
        new Float32Array(weights_array_buffer, 31180, 864 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__248_0 = builder.conv2d(
        var_re_lu_14_0, const_fold_opt__394_transposed,
        {
            bias: var_depthwise_conv2d_8_y_0, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 24, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_15_0 = builder.relu(var_Conv__248_0);

    const var_add_7__xeno_compat__1_0 = builder.add(var_re_lu_14_0, var_re_lu_15_0);

    const var_315 = builder.constant(
        {dataType: 'float32', shape: [8]},
        new Float32Array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    );

    const var_290 = builder.resample2d(
        var_add_7__xeno_compat__1_0,
        {
            mode: 'linear', scales: [2.0, 2.0], sizes: undefined, axes: [1, 2]
        }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_36_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 24]},
        new Float32Array(weights_array_buffer, 111852, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__249_0 = builder.conv2d(
        var_290, var_Conv2D_36_filter_0_transposed,
        {
            bias: var_conv2d_35_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_8__xeno_compat__1_0 = builder.add(var_Conv__167_0, var_Conv__249_0);

    const var_global_average_pooling2d_8_0 = builder.averagePool2d(
        var_add_8__xeno_compat__1_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_37_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 110828, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__252_0 = builder.conv2d(
        var_global_average_pooling2d_8_0, var_Conv2D_37_filter_0_transposed,
        {
            bias: var_conv2d_36_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_16_0 = builder.relu(var_Conv__252_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_38_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 109804, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__253_0 = builder.conv2d(
        var_re_lu_16_0, var_Conv2D_38_filter_0_transposed,
        {
            bias: var_conv2d_37_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_8_0 = builder.sigmoid(var_Conv__253_0);

    const var_multiply_8_0 = builder.mul(var_Conv__167_0, var_activation_8_0);

    const var_add_9__xeno_compat__1_0 = builder.add(var_multiply_8_0, var_Conv__249_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_39_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 108780, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__254_0 = builder.conv2d(
        var_add_9__xeno_compat__1_0, var_Conv2D_39_filter_0_transposed,
        {
            bias: var_conv2d_38_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_17_0 = builder.relu(var_Conv__254_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__391_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 3, 3, 16]},
        new Float32Array(weights_array_buffer, 32044, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__257_0 = builder.conv2d(
        var_re_lu_17_0, const_fold_opt__391_transposed,
        {
            bias: var_depthwise_conv2d_9_y_0, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 16, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_18_0 = builder.relu(var_Conv__257_0);

    const var_add_10__xeno_compat__1_0 = builder.add(var_re_lu_17_0, var_re_lu_18_0);

    const var_316 = builder.constant(
        {dataType: 'float32', shape: [8]},
        new Float32Array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    );

    const var_291 = builder.resample2d(
        var_add_10__xeno_compat__1_0,
        {
            mode: 'linear', scales: [2.0, 2.0], sizes: undefined, axes: [1, 2]
        }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_40_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 107244, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__258_0 = builder.conv2d(
        var_291, var_Conv2D_40_filter_0_transposed,
        {
            bias: var_conv2d_39_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_add_11__xeno_compat__1_0 = builder.add(var_mul_1_0, var_Conv__258_0);

    const var_global_average_pooling2d_9_0 = builder.averagePool2d(
        var_add_11__xeno_compat__1_0, { layout: 'nhwc' }
    );

    // Re-create constant operand from transposed weights.

    const var_Conv2D_41_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 106220, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__261_0 = builder.conv2d(
        var_global_average_pooling2d_9_0, var_Conv2D_41_filter_0_transposed,
        {
            bias: var_conv2d_40_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_19_0 = builder.relu(var_Conv__261_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_42_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 105196, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__262_0 = builder.conv2d(
        var_re_lu_19_0, var_Conv2D_42_filter_0_transposed,
        {
            bias: var_conv2d_41_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_activation_9_0 = builder.sigmoid(var_Conv__262_0);

    const var_multiply_9_0 = builder.mul(var_mul_1_0, var_activation_9_0);

    const var_add_12__xeno_compat__1_0 = builder.add(var_multiply_9_0, var_Conv__258_0);

    // Re-create constant operand from transposed weights.

    const var_Conv2D_43_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [16, 1, 1, 16]},
        new Float32Array(weights_array_buffer, 104172, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__263_0 = builder.conv2d(
        var_add_12__xeno_compat__1_0, var_Conv2D_43_filter_0_transposed,
        {
            bias: var_conv2d_42_1_y_0, strides: [1, 1], padding: undefined, dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_20_0 = builder.relu(var_Conv__263_0);

    // Re-create constant operand from transposed weights.

    const const_fold_opt__389_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 3, 3, 16]},
        new Float32Array(weights_array_buffer, 32620, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_Conv__266_0 = builder.conv2d(
        var_re_lu_20_0, const_fold_opt__389_transposed,
        {
            bias: var_depthwise_conv2d_10_y_0, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 16, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_re_lu_21_0 = builder.relu(var_Conv__266_0);

    const var_add_13__xeno_compat__1_0 = builder.add(var_re_lu_20_0, var_re_lu_21_0);

    // Re-create constant operand from transposed weights.

    const var_conv2d_transpose_filter_0_transposed = builder.constant(
        {dataType: 'float32', shape: [1, 2, 2, 16]},
        new Float32Array(weights_array_buffer, 3140, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2d_transpose_0 = builder.convTranspose2d(
        var_add_13__xeno_compat__1_0, var_conv2d_transpose_filter_0_transposed,
        {
            bias: undefined, strides: [2, 2], padding: undefined, dilations: [1, 1], groups: undefined, outputSizes: [144, 256], filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_conv2d_transpose_add_0 = builder.add(var_conv2d_transpose_0, const_fold_opt__402);

    const var_segment_back_raw_output___4_0 = builder.sigmoid(var_conv2d_transpose_add_0);

    const segment_back = builder.reshape(
        var_segment_back_raw_output___4_0,
        (() => {
        const shape = [1, 144, 256, 1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var_segment_back_raw_output___4_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    // Build graph with output operands.

    // Transpose output from NHWC TO NCHW.

    const segment_back_nchw = builder.transpose(
        segment_back,
        { permutation: [0, 3, 1, 2] }
    );

    this.graph_ = await builder.build({'segment_back': segment_back_nchw});

    // Create graph output tensors.

    this.outputTensors_['segment_back'] = await this.context_.createTensor(
        {dataType: segment_back_nchw.dataType, shape: segment_back_nchw.shape, readable: true}
    );

  }

  async run(inputs) {

    // Set input buffers to input tensors using writeTensor (sync)

    for (const name in inputs) {

      if (!(name in this.inputTensors_)) throw new Error(`Unknown input: ${name}`);

      this.context_.writeTensor(this.inputTensors_[name], inputs[name]);

    }

    // Compute the graph

    await this.context_.dispatch(this.graph_, this.inputTensors_, this.outputTensors_);

    // Read output tensors to buffers using readTensor (async)

    const outputs = {};

    for (const name in this.outputTensors_) {

      const tensor = this.outputTensors_[name];

      const buffer = await this.context_.readTensor(tensor);

      let typedArrayCtor;

      switch (tensor.dataType) {

        case 'float32': typedArrayCtor = Float32Array; break;

        case 'float64': typedArrayCtor = Float64Array; break;

        case 'int32': typedArrayCtor = Int32Array; break;

        case 'uint8': typedArrayCtor = Uint8Array; break;

        case 'int8': typedArrayCtor = Int8Array; break;

        case 'uint16': typedArrayCtor = Uint16Array; break;

        case 'int16': typedArrayCtor = Int16Array; break;

        case 'uint32': typedArrayCtor = Uint32Array; break;

        case 'int64': typedArrayCtor = BigInt64Array; break;

        case 'uint64': typedArrayCtor = BigUint64Array; break;

        default: throw new Error(`Unhandled tensor dataType: ${tensor.dataType}`);

      }

      outputs[name] = new typedArrayCtor(buffer);

    }

    return outputs;

  }

}