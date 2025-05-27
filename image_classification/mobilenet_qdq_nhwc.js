export class MobilenetQdqNhwc {

  constructor() {

    this.graph_ = null;

    this.context_ = null;

    this.inputTensors_ = {};

    this.outputTensors_ = {};

  }



  async build(contextOptions) {

    // Load weights ArrayBuffer from mobilenet_qdq_nhwc.bin
    async function loadWeightsArrayBuffer() {
        const response = await fetch('mobilenet_qdq_nhwc.bin');
        if (!response.ok) {
            throw new Error('Failed to fetch weights: ' + response.statusText);
        }
        return await response.arrayBuffer();
    }

    const weights_array_buffer = await loadWeightsArrayBuffer();

    this.context_ = await navigator.ml.createContext(contextOptions);
    const builder = new MLGraphBuilder(this.context_);


    // Create graph constant operands.

    const var_471 = builder.constant(
        {dataType: 'int64', shape: [2]},
        new BigInt64Array(weights_array_buffer, 0, 16 / BigInt64Array.BYTES_PER_ELEMENT)
    );

    const var_classifier_1_bias = builder.constant(
        {dataType: 'float32', shape: [1000]},
        new Float32Array(weights_array_buffer, 16, 4000 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_611_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 4016, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const input_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([114])
    );

    const input_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.018658448])
    );

    const var_475_quantized = builder.constant(
        {dataType: 'int8', shape: [32, 3, 3, 3]},
        new Int8Array(weights_array_buffer, 4020, 864 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_475_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0028556427])
    );

    const var_475_zero_point = builder.constant(
        {dataType: 'int8', shape: []},
        new Int8Array([0])
    );

    const var_474_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([0])
    );

    const var_474_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.017607756])
    );

    const var_476_quantized = builder.constant(
        {dataType: 'int32', shape: [32]},
        new Int32Array(weights_array_buffer, 4884, 128 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_476_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 5012, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_classifier_1_weight_quantized = builder.constant(
        {dataType: 'int8', shape: [1280, 1000]},
        new Int8Array(weights_array_buffer, 5016, 1280000 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_478_quantized = builder.constant(
        {dataType: 'int8', shape: [32, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1285016, 288 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_478_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.11868368])
    );

    const var_classifier_1_weight_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0026049719])
    );

    const var_479_quantized = builder.constant(
        {dataType: 'int32', shape: [32]},
        new Int32Array(weights_array_buffer, 1285304, 128 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_479_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1285432, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_464_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.019110054])
    );

    const var_481_quantized = builder.constant(
        {dataType: 'int8', shape: [16, 32, 1, 1]},
        new Int8Array(weights_array_buffer, 1285436, 512 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_481_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0081530325])
    );

    const var_480_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([119])
    );

    const var_480_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.06564797])
    );

    const var_482_quantized = builder.constant(
        {dataType: 'int32', shape: [16]},
        new Int32Array(weights_array_buffer, 1285948, 64 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_482_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1286012, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_484_quantized = builder.constant(
        {dataType: 'int8', shape: [96, 16, 1, 1]},
        new Int8Array(weights_array_buffer, 1286016, 1536 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_484_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.004745323])
    );

    const var_625_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.00507597])
    );

    const var_485_quantized = builder.constant(
        {dataType: 'int32', shape: [96]},
        new Int32Array(weights_array_buffer, 1287552, 384 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_485_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1287936, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_629_quantized = builder.constant(
        {dataType: 'int32', shape: [1280]},
        new Int32Array(weights_array_buffer, 1287940, 5120 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_487_quantized = builder.constant(
        {dataType: 'int8', shape: [96, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1293060, 864 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_487_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.05006342])
    );

    const var_488_quantized = builder.constant(
        {dataType: 'int32', shape: [96]},
        new Int32Array(weights_array_buffer, 1293924, 384 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_488_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1294308, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_490_quantized = builder.constant(
        {dataType: 'int8', shape: [24, 96, 1, 1]},
        new Int8Array(weights_array_buffer, 1294312, 2304 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_490_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.005946148])
    );

    const var_489_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([107])
    );

    const var_489_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.03971019])
    );

    const var_491_quantized = builder.constant(
        {dataType: 'int32', shape: [24]},
        new Int32Array(weights_array_buffer, 1296616, 96 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_491_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1296712, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_629_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1296716, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_493_quantized = builder.constant(
        {dataType: 'int8', shape: [144, 24, 1, 1]},
        new Int8Array(weights_array_buffer, 1296720, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_493_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0026336643])
    );

    const var_492_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.009025857])
    );

    const var_494_quantized = builder.constant(
        {dataType: 'int32', shape: [144]},
        new Int32Array(weights_array_buffer, 1300176, 576 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_494_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1300752, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_628_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.013814686])
    );

    const var_496_quantized = builder.constant(
        {dataType: 'int8', shape: [144, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1300756, 1296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_496_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.037832465])
    );

    const var_495_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.019913742])
    );

    const var_497_quantized = builder.constant(
        {dataType: 'int32', shape: [144]},
        new Int32Array(weights_array_buffer, 1302052, 576 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_497_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1302628, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_628_quantized = builder.constant(
        {dataType: 'int8', shape: [1280, 320, 1, 1]},
        new Int8Array(weights_array_buffer, 1302632, 409600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_499_quantized = builder.constant(
        {dataType: 'int8', shape: [24, 144, 1, 1]},
        new Int8Array(weights_array_buffer, 1712232, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_499_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.009542276])
    );

    const var_498_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([127])
    );

    const var_498_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.04609981])
    );

    const var_500_quantized = builder.constant(
        {dataType: 'int32', shape: [24]},
        new Int32Array(weights_array_buffer, 1715688, 96 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_500_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1715784, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_624_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.014525251])
    );

    const var_339_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([125])
    );

    const var_339_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.057971675])
    );

    const var_502_quantized = builder.constant(
        {dataType: 'int8', shape: [144, 24, 1, 1]},
        new Int8Array(weights_array_buffer, 1715788, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_502_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0024801705])
    );

    const var_501_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.013438906])
    );

    const var_503_quantized = builder.constant(
        {dataType: 'int32', shape: [144]},
        new Int32Array(weights_array_buffer, 1719244, 576 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_503_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1719820, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_626_quantized = builder.constant(
        {dataType: 'int32', shape: [320]},
        new Int32Array(weights_array_buffer, 1719824, 1280 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_505_quantized = builder.constant(
        {dataType: 'int8', shape: [144, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1721104, 1296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_505_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.046499304])
    );

    const var_504_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.014856177])
    );

    const var_506_quantized = builder.constant(
        {dataType: 'int32', shape: [144]},
        new Int32Array(weights_array_buffer, 1722400, 576 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_506_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1722976, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_508_quantized = builder.constant(
        {dataType: 'int8', shape: [32, 144, 1, 1]},
        new Int8Array(weights_array_buffer, 1722980, 4608 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_508_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.006835098])
    );

    const var_507_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([132])
    );

    const var_507_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.035562083])
    );

    const var_509_quantized = builder.constant(
        {dataType: 'int32', shape: [32]},
        new Int32Array(weights_array_buffer, 1727588, 128 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_509_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1727716, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_623_zero_point = builder.constant(
        {dataType: 'int32', shape: [1]},
        new Int32Array(weights_array_buffer, 1727720, 4 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_511_quantized = builder.constant(
        {dataType: 'int8', shape: [192, 32, 1, 1]},
        new Int8Array(weights_array_buffer, 1727724, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_511_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0015136111])
    );

    const var_510_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.008627743])
    );

    const var_512_quantized = builder.constant(
        {dataType: 'int32', shape: [192]},
        new Int32Array(weights_array_buffer, 1733868, 768 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_512_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1734636, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_623_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1734640, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_514_quantized = builder.constant(
        {dataType: 'int8', shape: [192, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1734644, 1728 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_514_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.05171833])
    );

    const var_513_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.012746858])
    );

    const var_515_quantized = builder.constant(
        {dataType: 'int32', shape: [192]},
        new Int32Array(weights_array_buffer, 1736372, 768 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_515_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1737140, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_623_quantized = builder.constant(
        {dataType: 'int32', shape: [960]},
        new Int32Array(weights_array_buffer, 1737144, 3840 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_517_quantized = builder.constant(
        {dataType: 'int8', shape: [32, 192, 1, 1]},
        new Int8Array(weights_array_buffer, 1740984, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_517_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0065188585])
    );

    const var_516_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([128])
    );

    const var_516_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.028934278])
    );

    const var_518_quantized = builder.constant(
        {dataType: 'int32', shape: [32]},
        new Int32Array(weights_array_buffer, 1747128, 128 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_518_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1747256, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_621_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.008207709])
    );

    const var_356_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([108])
    );

    const var_356_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.04353522])
    );

    const var_520_quantized = builder.constant(
        {dataType: 'int8', shape: [192, 32, 1, 1]},
        new Int8Array(weights_array_buffer, 1747260, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_520_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0012169179])
    );

    const var_519_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.007658489])
    );

    const var_521_quantized = builder.constant(
        {dataType: 'int32', shape: [192]},
        new Int32Array(weights_array_buffer, 1753404, 768 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_521_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1754172, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_626_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1754176, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_523_quantized = builder.constant(
        {dataType: 'int8', shape: [192, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1754180, 1728 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_523_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.039709866])
    );

    const var_522_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.010573713])
    );

    const var_524_quantized = builder.constant(
        {dataType: 'int32', shape: [192]},
        new Int32Array(weights_array_buffer, 1755908, 768 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_524_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1756676, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_526_quantized = builder.constant(
        {dataType: 'int8', shape: [32, 192, 1, 1]},
        new Int8Array(weights_array_buffer, 1756680, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_526_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0074411356])
    );

    const var_525_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([126])
    );

    const var_525_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.026068475])
    );

    const var_527_quantized = builder.constant(
        {dataType: 'int32', shape: [32]},
        new Int32Array(weights_array_buffer, 1762824, 128 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_527_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1762952, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_365_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([120])
    );

    const var_365_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.049828876])
    );

    const var_529_quantized = builder.constant(
        {dataType: 'int8', shape: [192, 32, 1, 1]},
        new Int8Array(weights_array_buffer, 1762956, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_529_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0017131757])
    );

    const var_528_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.011388029])
    );

    const var_530_quantized = builder.constant(
        {dataType: 'int32', shape: [192]},
        new Int32Array(weights_array_buffer, 1769100, 768 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_530_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1769868, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_622_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.07043054])
    );

    const var_532_quantized = builder.constant(
        {dataType: 'int8', shape: [192, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1769872, 1728 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_532_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.019297061])
    );

    const var_531_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.014233097])
    );

    const var_533_quantized = builder.constant(
        {dataType: 'int32', shape: [192]},
        new Int32Array(weights_array_buffer, 1771600, 768 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_533_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1772368, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_624_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([123])
    );

    const var_535_quantized = builder.constant(
        {dataType: 'int8', shape: [64, 192, 1, 1]},
        new Int8Array(weights_array_buffer, 1772372, 12288 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_535_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0054431446])
    );

    const var_534_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([143])
    );

    const var_534_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.028401868])
    );

    const var_536_quantized = builder.constant(
        {dataType: 'int32', shape: [64]},
        new Int32Array(weights_array_buffer, 1784660, 256 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_536_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1784916, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_620_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1784920, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_538_quantized = builder.constant(
        {dataType: 'int8', shape: [384, 64, 1, 1]},
        new Int8Array(weights_array_buffer, 1784924, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_538_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0011173795])
    );

    const var_537_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0043541207])
    );

    const var_539_quantized = builder.constant(
        {dataType: 'int32', shape: [384]},
        new Int32Array(weights_array_buffer, 1809500, 1536 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_539_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1811036, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_620_quantized = builder.constant(
        {dataType: 'int32', shape: [960]},
        new Int32Array(weights_array_buffer, 1811040, 3840 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_541_quantized = builder.constant(
        {dataType: 'int8', shape: [384, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1814880, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_541_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.043541186])
    );

    const var_540_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0067695524])
    );

    const var_542_quantized = builder.constant(
        {dataType: 'int32', shape: [384]},
        new Int32Array(weights_array_buffer, 1818336, 1536 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_542_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1819872, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_618_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.015018467])
    );

    const var_544_quantized = builder.constant(
        {dataType: 'int8', shape: [64, 384, 1, 1]},
        new Int8Array(weights_array_buffer, 1819876, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_544_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0054894034])
    );

    const var_543_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([130])
    );

    const var_543_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.019568179])
    );

    const var_545_quantized = builder.constant(
        {dataType: 'int32', shape: [64]},
        new Int32Array(weights_array_buffer, 1844452, 256 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_545_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1844708, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_382_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([136])
    );

    const var_382_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.029870728])
    );

    const var_547_quantized = builder.constant(
        {dataType: 'int8', shape: [384, 64, 1, 1]},
        new Int8Array(weights_array_buffer, 1844712, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_547_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0008492199])
    );

    const var_546_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.004029288])
    );

    const var_548_quantized = builder.constant(
        {dataType: 'int32', shape: [384]},
        new Int32Array(weights_array_buffer, 1869288, 1536 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_548_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1870824, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_619_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0010381987])
    );

    const var_550_quantized = builder.constant(
        {dataType: 'int8', shape: [384, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 1870828, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_550_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.088971674])
    );

    const var_549_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.006989649])
    );

    const var_551_quantized = builder.constant(
        {dataType: 'int32', shape: [384]},
        new Int32Array(weights_array_buffer, 1874284, 1536 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_551_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 1875820, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_619_quantized = builder.constant(
        {dataType: 'int8', shape: [960, 160, 1, 1]},
        new Int8Array(weights_array_buffer, 1875824, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_553_quantized = builder.constant(
        {dataType: 'int8', shape: [64, 384, 1, 1]},
        new Int8Array(weights_array_buffer, 2029424, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_553_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0055298656])
    );

    const var_552_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.016850423])
    );

    const var_554_quantized = builder.constant(
        {dataType: 'int32', shape: [64]},
        new Int32Array(weights_array_buffer, 2054000, 256 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_554_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2054256, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const output_MatMul_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([76])
    );

    const var_391_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.031078013])
    );

    const var_556_quantized = builder.constant(
        {dataType: 'int8', shape: [384, 64, 1, 1]},
        new Int8Array(weights_array_buffer, 2054260, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_556_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0008293222])
    );

    const var_555_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.005010089])
    );

    const var_557_quantized = builder.constant(
        {dataType: 'int32', shape: [384]},
        new Int32Array(weights_array_buffer, 2078836, 1536 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_557_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2080372, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_452_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.23722179])
    );

    const var_559_quantized = builder.constant(
        {dataType: 'int8', shape: [384, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 2080376, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_559_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.039976027])
    );

    const var_558_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.021815652])
    );

    const var_560_quantized = builder.constant(
        {dataType: 'int32', shape: [384]},
        new Int32Array(weights_array_buffer, 2083832, 1536 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_560_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2085368, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_622_quantized = builder.constant(
        {dataType: 'int8', shape: [960, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 2085372, 8640 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_562_quantized = builder.constant(
        {dataType: 'int8', shape: [64, 384, 1, 1]},
        new Int8Array(weights_array_buffer, 2094012, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_562_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.006337489])
    );

    const var_561_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([160])
    );

    const var_561_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.033086453])
    );

    const var_563_quantized = builder.constant(
        {dataType: 'int32', shape: [64]},
        new Int32Array(weights_array_buffer, 2118588, 256 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_563_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2118844, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_617_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2118848, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_400_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([139])
    );

    const var_400_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.03736598])
    );

    const var_565_quantized = builder.constant(
        {dataType: 'int8', shape: [384, 64, 1, 1]},
        new Int8Array(weights_array_buffer, 2118852, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_565_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0011116444])
    );

    const var_564_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.006757779])
    );

    const var_566_quantized = builder.constant(
        {dataType: 'int32', shape: [384]},
        new Int32Array(weights_array_buffer, 2143428, 1536 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_566_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2144964, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_617_quantized = builder.constant(
        {dataType: 'int32', shape: [160]},
        new Int32Array(weights_array_buffer, 2144968, 640 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_568_quantized = builder.constant(
        {dataType: 'int8', shape: [384, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 2145608, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_568_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.051308464])
    );

    const var_567_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.012513305])
    );

    const var_569_quantized = builder.constant(
        {dataType: 'int32', shape: [384]},
        new Int32Array(weights_array_buffer, 2149064, 1536 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_569_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2150600, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_615_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([131])
    );

    const var_571_quantized = builder.constant(
        {dataType: 'int8', shape: [96, 384, 1, 1]},
        new Int8Array(weights_array_buffer, 2150604, 36864 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_571_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.00487904])
    );

    const var_570_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.024123587])
    );

    const var_572_quantized = builder.constant(
        {dataType: 'int32', shape: [96]},
        new Int32Array(weights_array_buffer, 2187468, 384 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_572_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2187852, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_574_quantized = builder.constant(
        {dataType: 'int8', shape: [576, 96, 1, 1]},
        new Int8Array(weights_array_buffer, 2187856, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_574_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0013143878])
    );

    const var_573_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0067103086])
    );

    const var_575_quantized = builder.constant(
        {dataType: 'int32', shape: [576]},
        new Int32Array(weights_array_buffer, 2243152, 2304 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_575_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2245456, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_615_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.1445715])
    );

    const var_577_quantized = builder.constant(
        {dataType: 'int8', shape: [576, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 2245460, 5184 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_577_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.07777221])
    );

    const var_576_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.023529412])
    );

    const var_578_quantized = builder.constant(
        {dataType: 'int32', shape: [576]},
        new Int32Array(weights_array_buffer, 2250644, 2304 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_578_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2252948, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_616_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.004802136])
    );

    const var_580_quantized = builder.constant(
        {dataType: 'int8', shape: [96, 576, 1, 1]},
        new Int8Array(weights_array_buffer, 2252952, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_580_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.003954809])
    );

    const var_579_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([115])
    );

    const var_579_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.02798538])
    );

    const var_581_quantized = builder.constant(
        {dataType: 'int32', shape: [96]},
        new Int32Array(weights_array_buffer, 2308248, 384 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_581_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2308632, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_616_quantized = builder.constant(
        {dataType: 'int8', shape: [160, 960, 1, 1]},
        new Int8Array(weights_array_buffer, 2308636, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_417_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([109])
    );

    const var_417_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.031824797])
    );

    const var_583_quantized = builder.constant(
        {dataType: 'int8', shape: [576, 96, 1, 1]},
        new Int8Array(weights_array_buffer, 2462236, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_583_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0017125434])
    );

    const var_582_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.018577984])
    );

    const var_584_quantized = builder.constant(
        {dataType: 'int32', shape: [576]},
        new Int32Array(weights_array_buffer, 2517532, 2304 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_584_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2519836, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_452_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([110])
    );

    const var_586_quantized = builder.constant(
        {dataType: 'int8', shape: [576, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 2519840, 5184 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_586_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.06750055])
    );

    const var_587_quantized = builder.constant(
        {dataType: 'int32', shape: [576]},
        new Int32Array(weights_array_buffer, 2525024, 2304 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_587_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2527328, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_589_quantized = builder.constant(
        {dataType: 'int8', shape: [96, 576, 1, 1]},
        new Int8Array(weights_array_buffer, 2527332, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_589_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.009840593])
    );

    const var_588_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([142])
    );

    const var_588_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.06331735])
    );

    const var_590_quantized = builder.constant(
        {dataType: 'int32', shape: [96]},
        new Int32Array(weights_array_buffer, 2582628, 384 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_590_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2583012, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_614_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2583016, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_426_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.07048993])
    );

    const var_592_quantized = builder.constant(
        {dataType: 'int8', shape: [576, 96, 1, 1]},
        new Int8Array(weights_array_buffer, 2583020, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_592_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0014183413])
    );

    const var_591_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.014397864])
    );

    const var_593_quantized = builder.constant(
        {dataType: 'int32', shape: [576]},
        new Int32Array(weights_array_buffer, 2638316, 2304 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_593_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2640620, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_614_quantized = builder.constant(
        {dataType: 'int32', shape: [960]},
        new Int32Array(weights_array_buffer, 2640624, 3840 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_595_quantized = builder.constant(
        {dataType: 'int8', shape: [576, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 2644464, 5184 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_595_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.018182425])
    );

    const var_613_quantized = builder.constant(
        {dataType: 'int8', shape: [960, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 2649648, 8640 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_596_quantized = builder.constant(
        {dataType: 'int32', shape: [576]},
        new Int32Array(weights_array_buffer, 2658288, 2304 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_596_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 2660592, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_625_quantized = builder.constant(
        {dataType: 'int8', shape: [320, 960, 1, 1]},
        new Int8Array(weights_array_buffer, 2660596, 307200 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_598_quantized = builder.constant(
        {dataType: 'int8', shape: [160, 576, 1, 1]},
        new Int8Array(weights_array_buffer, 2967796, 92160 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_598_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0025079844])
    );

    const var_597_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([149])
    );

    const var_597_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.06929498])
    );

    const var_599_quantized = builder.constant(
        {dataType: 'int32', shape: [160]},
        new Int32Array(weights_array_buffer, 3059956, 640 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_599_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 3060596, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_601_quantized = builder.constant(
        {dataType: 'int8', shape: [960, 160, 1, 1]},
        new Int8Array(weights_array_buffer, 3060600, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_601_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0029125263])
    );

    const var_600_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.020971533])
    );

    const var_602_quantized = builder.constant(
        {dataType: 'int32', shape: [960]},
        new Int32Array(weights_array_buffer, 3214200, 3840 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_602_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 3218040, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const output_MatMul_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.16079935])
    );

    const var_604_quantized = builder.constant(
        {dataType: 'int8', shape: [960, 1, 3, 3]},
        new Int8Array(weights_array_buffer, 3218044, 8640 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_604_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.07850213])
    );

    const var_603_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.022908082])
    );

    const var_605_quantized = builder.constant(
        {dataType: 'int32', shape: [960]},
        new Int32Array(weights_array_buffer, 3226684, 3840 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_605_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 3230524, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_607_quantized = builder.constant(
        {dataType: 'int8', shape: [160, 960, 1, 1]},
        new Int8Array(weights_array_buffer, 3230528, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_607_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.002574469])
    );

    const var_606_zero_point = builder.constant(
        {dataType: 'uint8', shape: []},
        new Uint8Array([105])
    );

    const var_606_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.06415011])
    );

    const var_608_quantized = builder.constant(
        {dataType: 'int32', shape: [160]},
        new Int32Array(weights_array_buffer, 3384128, 640 / Int32Array.BYTES_PER_ELEMENT)
    );

    const var_608_scale = builder.constant(
        {dataType: 'float32', shape: [1]},
        new Float32Array(weights_array_buffer, 3384768, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_613_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.057705373])
    );

    const var_443_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.12168143])
    );

    const var_610_quantized = builder.constant(
        {dataType: 'int8', shape: [960, 160, 1, 1]},
        new Int8Array(weights_array_buffer, 3384772, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_610_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.0016836411])
    );

    const var_609_scale = builder.constant(
        {dataType: 'float32', shape: []},
        new Float32Array([0.02009347])
    );

    const var_611_quantized = builder.constant(
        {dataType: 'int32', shape: [960]},
        new Int32Array(weights_array_buffer, 3538372, 3840 / Int32Array.BYTES_PER_ELEMENT)
    );

    // Create graph input operands and tensors.

    // Transpose input from NCHW -> NHWC.

    const input = builder.transpose(
        builder.input('input', {dataType: 'float32', shape: [1, 3, 224, 224]}),
        { permutation: [0, 2, 3, 1] }
    );

    this.inputTensors_['input'] = await this.context_.createTensor(
        {dataType: 'float32', shape: [1, 3, 224, 224], writable: true}
    );

    // Create graph operators.

    const input_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.018658448])
    );

    const input_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([114])
    );

    const input_quantized = builder.quantizeLinear(
        input,
        input_scale_reshaped_1_1_1_1,
        input_zero_point_reshaped_1_1_1_1
    );

    const input_dequantized = builder.dequantizeLinear(
        input_quantized,
        input_scale_reshaped_1_1_1_1,
        input_zero_point_reshaped_1_1_1_1
    );

    const var_475_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0028556427])
    );

    const var_475_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'int8', shape: [1, 1, 1, 1]},
        new Int8Array([0])
    );

    const var_475_dequantized = builder.dequantizeLinear(
        var_475_quantized,
        var_475_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_476_dequantized = builder.dequantizeLinear(
        var_476_quantized,
        var_476_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_475_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [32, 3, 3, 3]},
        new Int8Array(weights_array_buffer, 4020, 864 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_475_dequantized_transposed = builder.dequantizeLinear(
        var_475_quantized_transposed,
        var_475_scale,
        var_475_zero_point
    );

    const var_474_QuantizeInput = builder.conv2d(
        input_dequantized, var_475_dequantized_transposed,
        {
            bias: var_476_dequantized, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_474_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.017607756])
    );

    const var_474_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([0])
    );

    const var_474_quantized = builder.quantizeLinear(
        var_474_QuantizeInput,
        var_474_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_474_Conv_2_dequantized = builder.dequantizeLinear(
        var_474_quantized,
        var_474_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_478_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.11868368])
    );

    const var_478_dequantized = builder.dequantizeLinear(
        var_478_quantized,
        var_478_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_479_dequantized = builder.dequantizeLinear(
        var_479_quantized,
        var_479_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_478_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 32]},
        new Int8Array(weights_array_buffer, 1285016, 288 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_478_dequantized_transposed = builder.dequantizeLinear(
        var_478_quantized_transposed,
        var_478_scale,
        var_475_zero_point
    );

    const var_477_QuantizeInput = builder.conv2d(
        var_474_Conv_2_dequantized, var_478_dequantized_transposed,
        {
            bias: var_479_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 32, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_576_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.023529412])
    );

    const var_477_quantized = builder.quantizeLinear(
        var_477_QuantizeInput,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_477_Conv_4_dequantized = builder.dequantizeLinear(
        var_477_quantized,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_481_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0081530325])
    );

    const var_481_dequantized = builder.dequantizeLinear(
        var_481_quantized,
        var_481_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_482_dequantized = builder.dequantizeLinear(
        var_482_quantized,
        var_482_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_481_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [16, 1, 1, 32]},
        new Int8Array(weights_array_buffer, 1285436, 512 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_481_dequantized_transposed = builder.dequantizeLinear(
        var_481_quantized_transposed,
        var_481_scale,
        var_475_zero_point
    );

    const var_480_QuantizeInput = builder.conv2d(
        var_477_Conv_4_dequantized, var_481_dequantized_transposed,
        {
            bias: var_482_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_480_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.06564797])
    );

    const var_480_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([119])
    );

    const var_480_quantized = builder.quantizeLinear(
        var_480_QuantizeInput,
        var_480_scale_reshaped_1_1_1_1,
        var_480_zero_point_reshaped_1_1_1_1
    );

    const var_480_Conv_5_dequantized = builder.dequantizeLinear(
        var_480_quantized,
        var_480_scale_reshaped_1_1_1_1,
        var_480_zero_point_reshaped_1_1_1_1
    );

    const var_484_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.004745323])
    );

    const var_484_dequantized = builder.dequantizeLinear(
        var_484_quantized,
        var_484_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_485_dequantized = builder.dequantizeLinear(
        var_485_quantized,
        var_485_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_484_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [96, 1, 1, 16]},
        new Int8Array(weights_array_buffer, 1286016, 1536 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_484_dequantized_transposed = builder.dequantizeLinear(
        var_484_quantized_transposed,
        var_484_scale,
        var_475_zero_point
    );

    const var_483_QuantizeInput = builder.conv2d(
        var_480_Conv_5_dequantized, var_484_dequantized_transposed,
        {
            bias: var_485_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_483_quantized = builder.quantizeLinear(
        var_483_QuantizeInput,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_483_Conv_7_dequantized = builder.dequantizeLinear(
        var_483_quantized,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_487_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.05006342])
    );

    const var_487_dequantized = builder.dequantizeLinear(
        var_487_quantized,
        var_487_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_488_dequantized = builder.dequantizeLinear(
        var_488_quantized,
        var_488_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_487_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 96]},
        new Int8Array(weights_array_buffer, 1293060, 864 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_487_dequantized_transposed = builder.dequantizeLinear(
        var_487_quantized_transposed,
        var_487_scale,
        var_475_zero_point
    );

    const var_486_QuantizeInput = builder.conv2d(
        var_483_Conv_7_dequantized, var_487_dequantized_transposed,
        {
            bias: var_488_dequantized, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 96, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_486_quantized = builder.quantizeLinear(
        var_486_QuantizeInput,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_486_Conv_9_dequantized = builder.dequantizeLinear(
        var_486_quantized,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_490_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.005946148])
    );

    const var_490_dequantized = builder.dequantizeLinear(
        var_490_quantized,
        var_490_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_491_dequantized = builder.dequantizeLinear(
        var_491_quantized,
        var_491_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_490_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [24, 1, 1, 96]},
        new Int8Array(weights_array_buffer, 1294312, 2304 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_490_dequantized_transposed = builder.dequantizeLinear(
        var_490_quantized_transposed,
        var_490_scale,
        var_475_zero_point
    );

    const var_489_QuantizeInput = builder.conv2d(
        var_486_Conv_9_dequantized, var_490_dequantized_transposed,
        {
            bias: var_491_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_489_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.03971019])
    );

    const var_489_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([107])
    );

    const var_489_quantized = builder.quantizeLinear(
        var_489_QuantizeInput,
        var_489_scale_reshaped_1_1_1_1,
        var_489_zero_point_reshaped_1_1_1_1
    );

    const var_489_duplicated = builder.dequantizeLinear(
        var_489_quantized,
        var_489_scale_reshaped_1_1_1_1,
        var_489_zero_point_reshaped_1_1_1_1
    );

    const var_493_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0026336643])
    );

    const var_493_dequantized = builder.dequantizeLinear(
        var_493_quantized,
        var_493_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_494_dequantized = builder.dequantizeLinear(
        var_494_quantized,
        var_494_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_493_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [144, 1, 1, 24]},
        new Int8Array(weights_array_buffer, 1296720, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_493_dequantized_transposed = builder.dequantizeLinear(
        var_493_quantized_transposed,
        var_493_scale,
        var_475_zero_point
    );

    const var_492_QuantizeInput = builder.conv2d(
        var_489_duplicated, var_493_dequantized_transposed,
        {
            bias: var_494_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_492_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.009025857])
    );

    const var_492_quantized = builder.quantizeLinear(
        var_492_QuantizeInput,
        var_492_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_492_Conv_12_dequantized = builder.dequantizeLinear(
        var_492_quantized,
        var_492_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_496_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.037832465])
    );

    const var_496_dequantized = builder.dequantizeLinear(
        var_496_quantized,
        var_496_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_497_dequantized = builder.dequantizeLinear(
        var_497_quantized,
        var_497_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_496_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 144]},
        new Int8Array(weights_array_buffer, 1300756, 1296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_496_dequantized_transposed = builder.dequantizeLinear(
        var_496_quantized_transposed,
        var_496_scale,
        var_475_zero_point
    );

    const var_495_QuantizeInput = builder.conv2d(
        var_492_Conv_12_dequantized, var_496_dequantized_transposed,
        {
            bias: var_497_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 144, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_495_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.019913742])
    );

    const var_495_quantized = builder.quantizeLinear(
        var_495_QuantizeInput,
        var_495_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_495_Conv_14_dequantized = builder.dequantizeLinear(
        var_495_quantized,
        var_495_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_499_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.009542276])
    );

    const var_499_dequantized = builder.dequantizeLinear(
        var_499_quantized,
        var_499_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_500_dequantized = builder.dequantizeLinear(
        var_500_quantized,
        var_500_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_499_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [24, 1, 1, 144]},
        new Int8Array(weights_array_buffer, 1712232, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_499_dequantized_transposed = builder.dequantizeLinear(
        var_499_quantized_transposed,
        var_499_scale,
        var_475_zero_point
    );

    const var_498_QuantizeInput = builder.conv2d(
        var_495_Conv_14_dequantized, var_499_dequantized_transposed,
        {
            bias: var_500_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_498_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.04609981])
    );

    const var_498_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([127])
    );

    const var_498_quantized = builder.quantizeLinear(
        var_498_QuantizeInput,
        var_498_scale_reshaped_1_1_1_1,
        var_498_zero_point_reshaped_1_1_1_1
    );

    const var_498 = builder.dequantizeLinear(
        var_498_quantized,
        var_498_scale_reshaped_1_1_1_1,
        var_498_zero_point_reshaped_1_1_1_1
    );

    const var_339 = builder.add(var_489_duplicated, var_498);

    const var_339_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.057971675])
    );

    const var_339_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([125])
    );

    const var_339_Conv_16_QuantizeLinear = builder.quantizeLinear(
        var_339,
        var_339_scale_reshaped_1_1_1_1,
        var_339_zero_point_reshaped_1_1_1_1
    );

    const var_339_Conv_16_dequantized = builder.dequantizeLinear(
        var_339_Conv_16_QuantizeLinear,
        var_339_scale_reshaped_1_1_1_1,
        var_339_zero_point_reshaped_1_1_1_1
    );

    const var_502_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0024801705])
    );

    const var_502_dequantized = builder.dequantizeLinear(
        var_502_quantized,
        var_502_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_503_dequantized = builder.dequantizeLinear(
        var_503_quantized,
        var_503_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_502_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [144, 1, 1, 24]},
        new Int8Array(weights_array_buffer, 1715788, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_502_dequantized_transposed = builder.dequantizeLinear(
        var_502_quantized_transposed,
        var_502_scale,
        var_475_zero_point
    );

    const var_501_QuantizeInput = builder.conv2d(
        var_339_Conv_16_dequantized, var_502_dequantized_transposed,
        {
            bias: var_503_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_501_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.013438906])
    );

    const var_501_quantized = builder.quantizeLinear(
        var_501_QuantizeInput,
        var_501_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_501_Conv_18_dequantized = builder.dequantizeLinear(
        var_501_quantized,
        var_501_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_505_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.046499304])
    );

    const var_505_dequantized = builder.dequantizeLinear(
        var_505_quantized,
        var_505_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_506_dequantized = builder.dequantizeLinear(
        var_506_quantized,
        var_506_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_505_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 144]},
        new Int8Array(weights_array_buffer, 1721104, 1296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_505_dequantized_transposed = builder.dequantizeLinear(
        var_505_quantized_transposed,
        var_505_scale,
        var_475_zero_point
    );

    const var_504_QuantizeInput = builder.conv2d(
        var_501_Conv_18_dequantized, var_505_dequantized_transposed,
        {
            bias: var_506_dequantized, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 144, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_504_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.014856177])
    );

    const var_504_quantized = builder.quantizeLinear(
        var_504_QuantizeInput,
        var_504_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_504_Conv_20_dequantized = builder.dequantizeLinear(
        var_504_quantized,
        var_504_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_508_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.006835098])
    );

    const var_508_dequantized = builder.dequantizeLinear(
        var_508_quantized,
        var_508_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_509_dequantized = builder.dequantizeLinear(
        var_509_quantized,
        var_509_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_508_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [32, 1, 1, 144]},
        new Int8Array(weights_array_buffer, 1722980, 4608 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_508_dequantized_transposed = builder.dequantizeLinear(
        var_508_quantized_transposed,
        var_508_scale,
        var_475_zero_point
    );

    const var_507_QuantizeInput = builder.conv2d(
        var_504_Conv_20_dequantized, var_508_dequantized_transposed,
        {
            bias: var_509_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_507_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.035562083])
    );

    const var_507_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([132])
    );

    const var_507_quantized = builder.quantizeLinear(
        var_507_QuantizeInput,
        var_507_scale_reshaped_1_1_1_1,
        var_507_zero_point_reshaped_1_1_1_1
    );

    const var_507_duplicated = builder.dequantizeLinear(
        var_507_quantized,
        var_507_scale_reshaped_1_1_1_1,
        var_507_zero_point_reshaped_1_1_1_1
    );

    const var_511_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0015136111])
    );

    const var_511_dequantized = builder.dequantizeLinear(
        var_511_quantized,
        var_511_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_512_dequantized = builder.dequantizeLinear(
        var_512_quantized,
        var_512_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_511_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [192, 1, 1, 32]},
        new Int8Array(weights_array_buffer, 1727724, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_511_dequantized_transposed = builder.dequantizeLinear(
        var_511_quantized_transposed,
        var_511_scale,
        var_475_zero_point
    );

    const var_510_QuantizeInput = builder.conv2d(
        var_507_duplicated, var_511_dequantized_transposed,
        {
            bias: var_512_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_510_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.008627743])
    );

    const var_510_quantized = builder.quantizeLinear(
        var_510_QuantizeInput,
        var_510_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_510_Conv_23_dequantized = builder.dequantizeLinear(
        var_510_quantized,
        var_510_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_514_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.05171833])
    );

    const var_514_dequantized = builder.dequantizeLinear(
        var_514_quantized,
        var_514_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_515_dequantized = builder.dequantizeLinear(
        var_515_quantized,
        var_515_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_514_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 192]},
        new Int8Array(weights_array_buffer, 1734644, 1728 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_514_dequantized_transposed = builder.dequantizeLinear(
        var_514_quantized_transposed,
        var_514_scale,
        var_475_zero_point
    );

    const var_513_QuantizeInput = builder.conv2d(
        var_510_Conv_23_dequantized, var_514_dequantized_transposed,
        {
            bias: var_515_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 192, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_513_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.012746858])
    );

    const var_513_quantized = builder.quantizeLinear(
        var_513_QuantizeInput,
        var_513_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_513_Conv_25_dequantized = builder.dequantizeLinear(
        var_513_quantized,
        var_513_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_517_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0065188585])
    );

    const var_517_dequantized = builder.dequantizeLinear(
        var_517_quantized,
        var_517_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_518_dequantized = builder.dequantizeLinear(
        var_518_quantized,
        var_518_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_517_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [32, 1, 1, 192]},
        new Int8Array(weights_array_buffer, 1740984, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_517_dequantized_transposed = builder.dequantizeLinear(
        var_517_quantized_transposed,
        var_517_scale,
        var_475_zero_point
    );

    const var_516_QuantizeInput = builder.conv2d(
        var_513_Conv_25_dequantized, var_517_dequantized_transposed,
        {
            bias: var_518_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_516_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.028934278])
    );

    const var_516_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([128])
    );

    const var_516_quantized = builder.quantizeLinear(
        var_516_QuantizeInput,
        var_516_scale_reshaped_1_1_1_1,
        var_516_zero_point_reshaped_1_1_1_1
    );

    const var_516 = builder.dequantizeLinear(
        var_516_quantized,
        var_516_scale_reshaped_1_1_1_1,
        var_516_zero_point_reshaped_1_1_1_1
    );

    const var_356 = builder.add(var_507_duplicated, var_516);

    const var_356_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.04353522])
    );

    const var_356_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([108])
    );

    const var_356_Conv_27_QuantizeLinear = builder.quantizeLinear(
        var_356,
        var_356_scale_reshaped_1_1_1_1,
        var_356_zero_point_reshaped_1_1_1_1
    );

    const var_356_Conv_27_dequantized = builder.dequantizeLinear(
        var_356_Conv_27_QuantizeLinear,
        var_356_scale_reshaped_1_1_1_1,
        var_356_zero_point_reshaped_1_1_1_1
    );

    const var_520_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0012169179])
    );

    const var_520_dequantized = builder.dequantizeLinear(
        var_520_quantized,
        var_520_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_521_dequantized = builder.dequantizeLinear(
        var_521_quantized,
        var_521_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_520_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [192, 1, 1, 32]},
        new Int8Array(weights_array_buffer, 1747260, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_520_dequantized_transposed = builder.dequantizeLinear(
        var_520_quantized_transposed,
        var_520_scale,
        var_475_zero_point
    );

    const var_519_QuantizeInput = builder.conv2d(
        var_356_Conv_27_dequantized, var_520_dequantized_transposed,
        {
            bias: var_521_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_519_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.007658489])
    );

    const var_519_quantized = builder.quantizeLinear(
        var_519_QuantizeInput,
        var_519_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_519_Conv_29_dequantized = builder.dequantizeLinear(
        var_519_quantized,
        var_519_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_523_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.039709866])
    );

    const var_523_dequantized = builder.dequantizeLinear(
        var_523_quantized,
        var_523_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_524_dequantized = builder.dequantizeLinear(
        var_524_quantized,
        var_524_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_523_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 192]},
        new Int8Array(weights_array_buffer, 1754180, 1728 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_523_dequantized_transposed = builder.dequantizeLinear(
        var_523_quantized_transposed,
        var_523_scale,
        var_475_zero_point
    );

    const var_522_QuantizeInput = builder.conv2d(
        var_519_Conv_29_dequantized, var_523_dequantized_transposed,
        {
            bias: var_524_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 192, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_522_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.010573713])
    );

    const var_522_quantized = builder.quantizeLinear(
        var_522_QuantizeInput,
        var_522_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_522_Conv_31_dequantized = builder.dequantizeLinear(
        var_522_quantized,
        var_522_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_526_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0074411356])
    );

    const var_526_dequantized = builder.dequantizeLinear(
        var_526_quantized,
        var_526_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_527_dequantized = builder.dequantizeLinear(
        var_527_quantized,
        var_527_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_526_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [32, 1, 1, 192]},
        new Int8Array(weights_array_buffer, 1756680, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_526_dequantized_transposed = builder.dequantizeLinear(
        var_526_quantized_transposed,
        var_526_scale,
        var_475_zero_point
    );

    const var_525_QuantizeInput = builder.conv2d(
        var_522_Conv_31_dequantized, var_526_dequantized_transposed,
        {
            bias: var_527_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_525_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.026068475])
    );

    const var_525_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([126])
    );

    const var_525_quantized = builder.quantizeLinear(
        var_525_QuantizeInput,
        var_525_scale_reshaped_1_1_1_1,
        var_525_zero_point_reshaped_1_1_1_1
    );

    const var_525 = builder.dequantizeLinear(
        var_525_quantized,
        var_525_scale_reshaped_1_1_1_1,
        var_525_zero_point_reshaped_1_1_1_1
    );

    const var_365 = builder.add(var_356, var_525);

    const var_365_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.049828876])
    );

    const var_365_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([120])
    );

    const var_365_Conv_33_QuantizeLinear = builder.quantizeLinear(
        var_365,
        var_365_scale_reshaped_1_1_1_1,
        var_365_zero_point_reshaped_1_1_1_1
    );

    const var_365_Conv_33_dequantized = builder.dequantizeLinear(
        var_365_Conv_33_QuantizeLinear,
        var_365_scale_reshaped_1_1_1_1,
        var_365_zero_point_reshaped_1_1_1_1
    );

    const var_529_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0017131757])
    );

    const var_529_dequantized = builder.dequantizeLinear(
        var_529_quantized,
        var_529_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_530_dequantized = builder.dequantizeLinear(
        var_530_quantized,
        var_530_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_529_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [192, 1, 1, 32]},
        new Int8Array(weights_array_buffer, 1762956, 6144 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_529_dequantized_transposed = builder.dequantizeLinear(
        var_529_quantized_transposed,
        var_529_scale,
        var_475_zero_point
    );

    const var_528_QuantizeInput = builder.conv2d(
        var_365_Conv_33_dequantized, var_529_dequantized_transposed,
        {
            bias: var_530_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_528_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.011388029])
    );

    const var_528_quantized = builder.quantizeLinear(
        var_528_QuantizeInput,
        var_528_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_528_Conv_35_dequantized = builder.dequantizeLinear(
        var_528_quantized,
        var_528_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_532_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.019297061])
    );

    const var_532_dequantized = builder.dequantizeLinear(
        var_532_quantized,
        var_532_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_533_dequantized = builder.dequantizeLinear(
        var_533_quantized,
        var_533_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_532_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 192]},
        new Int8Array(weights_array_buffer, 1769872, 1728 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_532_dequantized_transposed = builder.dequantizeLinear(
        var_532_quantized_transposed,
        var_532_scale,
        var_475_zero_point
    );

    const var_531_QuantizeInput = builder.conv2d(
        var_528_Conv_35_dequantized, var_532_dequantized_transposed,
        {
            bias: var_533_dequantized, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 192, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_531_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.014233097])
    );

    const var_531_quantized = builder.quantizeLinear(
        var_531_QuantizeInput,
        var_531_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_531_Conv_37_dequantized = builder.dequantizeLinear(
        var_531_quantized,
        var_531_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_535_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0054431446])
    );

    const var_535_dequantized = builder.dequantizeLinear(
        var_535_quantized,
        var_535_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_536_dequantized = builder.dequantizeLinear(
        var_536_quantized,
        var_536_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_535_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [64, 1, 1, 192]},
        new Int8Array(weights_array_buffer, 1772372, 12288 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_535_dequantized_transposed = builder.dequantizeLinear(
        var_535_quantized_transposed,
        var_535_scale,
        var_475_zero_point
    );

    const var_534_QuantizeInput = builder.conv2d(
        var_531_Conv_37_dequantized, var_535_dequantized_transposed,
        {
            bias: var_536_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_534_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.028401868])
    );

    const var_534_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([143])
    );

    const var_534_quantized = builder.quantizeLinear(
        var_534_QuantizeInput,
        var_534_scale_reshaped_1_1_1_1,
        var_534_zero_point_reshaped_1_1_1_1
    );

    const var_534_duplicated = builder.dequantizeLinear(
        var_534_quantized,
        var_534_scale_reshaped_1_1_1_1,
        var_534_zero_point_reshaped_1_1_1_1
    );

    const var_538_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0011173795])
    );

    const var_538_dequantized = builder.dequantizeLinear(
        var_538_quantized,
        var_538_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_539_dequantized = builder.dequantizeLinear(
        var_539_quantized,
        var_539_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_538_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [384, 1, 1, 64]},
        new Int8Array(weights_array_buffer, 1784924, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_538_dequantized_transposed = builder.dequantizeLinear(
        var_538_quantized_transposed,
        var_538_scale,
        var_475_zero_point
    );

    const var_537_QuantizeInput = builder.conv2d(
        var_534_duplicated, var_538_dequantized_transposed,
        {
            bias: var_539_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_537_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0043541207])
    );

    const var_537_quantized = builder.quantizeLinear(
        var_537_QuantizeInput,
        var_537_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_537_Conv_40_dequantized = builder.dequantizeLinear(
        var_537_quantized,
        var_537_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_541_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.043541186])
    );

    const var_541_dequantized = builder.dequantizeLinear(
        var_541_quantized,
        var_541_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_542_dequantized = builder.dequantizeLinear(
        var_542_quantized,
        var_542_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_541_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 384]},
        new Int8Array(weights_array_buffer, 1814880, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_541_dequantized_transposed = builder.dequantizeLinear(
        var_541_quantized_transposed,
        var_541_scale,
        var_475_zero_point
    );

    const var_540_QuantizeInput = builder.conv2d(
        var_537_Conv_40_dequantized, var_541_dequantized_transposed,
        {
            bias: var_542_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 384, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_540_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0067695524])
    );

    const var_540_quantized = builder.quantizeLinear(
        var_540_QuantizeInput,
        var_540_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_540_Conv_42_dequantized = builder.dequantizeLinear(
        var_540_quantized,
        var_540_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_544_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0054894034])
    );

    const var_544_dequantized = builder.dequantizeLinear(
        var_544_quantized,
        var_544_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_545_dequantized = builder.dequantizeLinear(
        var_545_quantized,
        var_545_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_544_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [64, 1, 1, 384]},
        new Int8Array(weights_array_buffer, 1819876, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_544_dequantized_transposed = builder.dequantizeLinear(
        var_544_quantized_transposed,
        var_544_scale,
        var_475_zero_point
    );

    const var_543_QuantizeInput = builder.conv2d(
        var_540_Conv_42_dequantized, var_544_dequantized_transposed,
        {
            bias: var_545_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_543_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.019568179])
    );

    const var_543_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([130])
    );

    const var_543_quantized = builder.quantizeLinear(
        var_543_QuantizeInput,
        var_543_scale_reshaped_1_1_1_1,
        var_543_zero_point_reshaped_1_1_1_1
    );

    const var_543 = builder.dequantizeLinear(
        var_543_quantized,
        var_543_scale_reshaped_1_1_1_1,
        var_543_zero_point_reshaped_1_1_1_1
    );

    const var_382 = builder.add(var_534_duplicated, var_543);

    const var_382_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.029870728])
    );

    const var_382_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([136])
    );

    const var_382_Conv_44_QuantizeLinear = builder.quantizeLinear(
        var_382,
        var_382_scale_reshaped_1_1_1_1,
        var_382_zero_point_reshaped_1_1_1_1
    );

    const var_382_Conv_44_dequantized = builder.dequantizeLinear(
        var_382_Conv_44_QuantizeLinear,
        var_382_scale_reshaped_1_1_1_1,
        var_382_zero_point_reshaped_1_1_1_1
    );

    const var_547_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0008492199])
    );

    const var_547_dequantized = builder.dequantizeLinear(
        var_547_quantized,
        var_547_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_548_dequantized = builder.dequantizeLinear(
        var_548_quantized,
        var_548_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_547_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [384, 1, 1, 64]},
        new Int8Array(weights_array_buffer, 1844712, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_547_dequantized_transposed = builder.dequantizeLinear(
        var_547_quantized_transposed,
        var_547_scale,
        var_475_zero_point
    );

    const var_546_QuantizeInput = builder.conv2d(
        var_382_Conv_44_dequantized, var_547_dequantized_transposed,
        {
            bias: var_548_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_546_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.004029288])
    );

    const var_546_quantized = builder.quantizeLinear(
        var_546_QuantizeInput,
        var_546_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_546_Conv_46_dequantized = builder.dequantizeLinear(
        var_546_quantized,
        var_546_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_550_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.088971674])
    );

    const var_550_dequantized = builder.dequantizeLinear(
        var_550_quantized,
        var_550_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_551_dequantized = builder.dequantizeLinear(
        var_551_quantized,
        var_551_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_550_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 384]},
        new Int8Array(weights_array_buffer, 1870828, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_550_dequantized_transposed = builder.dequantizeLinear(
        var_550_quantized_transposed,
        var_550_scale,
        var_475_zero_point
    );

    const var_549_QuantizeInput = builder.conv2d(
        var_546_Conv_46_dequantized, var_550_dequantized_transposed,
        {
            bias: var_551_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 384, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_549_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.006989649])
    );

    const var_549_quantized = builder.quantizeLinear(
        var_549_QuantizeInput,
        var_549_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_549_Conv_48_dequantized = builder.dequantizeLinear(
        var_549_quantized,
        var_549_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_553_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0055298656])
    );

    const var_553_dequantized = builder.dequantizeLinear(
        var_553_quantized,
        var_553_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_554_dequantized = builder.dequantizeLinear(
        var_554_quantized,
        var_554_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_553_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [64, 1, 1, 384]},
        new Int8Array(weights_array_buffer, 2029424, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_553_dequantized_transposed = builder.dequantizeLinear(
        var_553_quantized_transposed,
        var_553_scale,
        var_475_zero_point
    );

    const var_552_QuantizeInput = builder.conv2d(
        var_549_Conv_48_dequantized, var_553_dequantized_transposed,
        {
            bias: var_554_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_552_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.016850423])
    );

    const var_552_quantized = builder.quantizeLinear(
        var_552_QuantizeInput,
        var_552_scale_reshaped_1_1_1_1,
        var_543_zero_point_reshaped_1_1_1_1
    );

    const var_552 = builder.dequantizeLinear(
        var_552_quantized,
        var_552_scale_reshaped_1_1_1_1,
        var_543_zero_point_reshaped_1_1_1_1
    );

    const var_391 = builder.add(var_382, var_552);

    const var_391_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.031078013])
    );

    const var_391_Conv_50_QuantizeLinear = builder.quantizeLinear(
        var_391,
        var_391_scale_reshaped_1_1_1_1,
        var_525_zero_point_reshaped_1_1_1_1
    );

    const var_391_Conv_50_dequantized = builder.dequantizeLinear(
        var_391_Conv_50_QuantizeLinear,
        var_391_scale_reshaped_1_1_1_1,
        var_525_zero_point_reshaped_1_1_1_1
    );

    const var_556_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0008293222])
    );

    const var_556_dequantized = builder.dequantizeLinear(
        var_556_quantized,
        var_556_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_557_dequantized = builder.dequantizeLinear(
        var_557_quantized,
        var_557_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_556_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [384, 1, 1, 64]},
        new Int8Array(weights_array_buffer, 2054260, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_556_dequantized_transposed = builder.dequantizeLinear(
        var_556_quantized_transposed,
        var_556_scale,
        var_475_zero_point
    );

    const var_555_QuantizeInput = builder.conv2d(
        var_391_Conv_50_dequantized, var_556_dequantized_transposed,
        {
            bias: var_557_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_555_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.005010089])
    );

    const var_555_quantized = builder.quantizeLinear(
        var_555_QuantizeInput,
        var_555_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_555_Conv_52_dequantized = builder.dequantizeLinear(
        var_555_quantized,
        var_555_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_559_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.039976027])
    );

    const var_559_dequantized = builder.dequantizeLinear(
        var_559_quantized,
        var_559_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_560_dequantized = builder.dequantizeLinear(
        var_560_quantized,
        var_560_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_559_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 384]},
        new Int8Array(weights_array_buffer, 2080376, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_559_dequantized_transposed = builder.dequantizeLinear(
        var_559_quantized_transposed,
        var_559_scale,
        var_475_zero_point
    );

    const var_558_QuantizeInput = builder.conv2d(
        var_555_Conv_52_dequantized, var_559_dequantized_transposed,
        {
            bias: var_560_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 384, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_558_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.021815652])
    );

    const var_558_quantized = builder.quantizeLinear(
        var_558_QuantizeInput,
        var_558_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_558_Conv_54_dequantized = builder.dequantizeLinear(
        var_558_quantized,
        var_558_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_562_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.006337489])
    );

    const var_562_dequantized = builder.dequantizeLinear(
        var_562_quantized,
        var_562_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_563_dequantized = builder.dequantizeLinear(
        var_563_quantized,
        var_563_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_562_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [64, 1, 1, 384]},
        new Int8Array(weights_array_buffer, 2094012, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_562_dequantized_transposed = builder.dequantizeLinear(
        var_562_quantized_transposed,
        var_562_scale,
        var_475_zero_point
    );

    const var_561_QuantizeInput = builder.conv2d(
        var_558_Conv_54_dequantized, var_562_dequantized_transposed,
        {
            bias: var_563_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_561_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.033086453])
    );

    const var_561_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([160])
    );

    const var_561_quantized = builder.quantizeLinear(
        var_561_QuantizeInput,
        var_561_scale_reshaped_1_1_1_1,
        var_561_zero_point_reshaped_1_1_1_1
    );

    const var_561 = builder.dequantizeLinear(
        var_561_quantized,
        var_561_scale_reshaped_1_1_1_1,
        var_561_zero_point_reshaped_1_1_1_1
    );

    const var_400 = builder.add(var_391, var_561);

    const var_400_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.03736598])
    );

    const var_400_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([139])
    );

    const var_400_Conv_56_QuantizeLinear = builder.quantizeLinear(
        var_400,
        var_400_scale_reshaped_1_1_1_1,
        var_400_zero_point_reshaped_1_1_1_1
    );

    const var_400_Conv_56_dequantized = builder.dequantizeLinear(
        var_400_Conv_56_QuantizeLinear,
        var_400_scale_reshaped_1_1_1_1,
        var_400_zero_point_reshaped_1_1_1_1
    );

    const var_565_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0011116444])
    );

    const var_565_dequantized = builder.dequantizeLinear(
        var_565_quantized,
        var_565_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_566_dequantized = builder.dequantizeLinear(
        var_566_quantized,
        var_566_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_565_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [384, 1, 1, 64]},
        new Int8Array(weights_array_buffer, 2118852, 24576 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_565_dequantized_transposed = builder.dequantizeLinear(
        var_565_quantized_transposed,
        var_565_scale,
        var_475_zero_point
    );

    const var_564_QuantizeInput = builder.conv2d(
        var_400_Conv_56_dequantized, var_565_dequantized_transposed,
        {
            bias: var_566_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_564_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.006757779])
    );

    const var_564_quantized = builder.quantizeLinear(
        var_564_QuantizeInput,
        var_564_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_564_Conv_58_dequantized = builder.dequantizeLinear(
        var_564_quantized,
        var_564_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_568_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.051308464])
    );

    const var_568_dequantized = builder.dequantizeLinear(
        var_568_quantized,
        var_568_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_569_dequantized = builder.dequantizeLinear(
        var_569_quantized,
        var_569_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_568_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 384]},
        new Int8Array(weights_array_buffer, 2145608, 3456 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_568_dequantized_transposed = builder.dequantizeLinear(
        var_568_quantized_transposed,
        var_568_scale,
        var_475_zero_point
    );

    const var_567_QuantizeInput = builder.conv2d(
        var_564_Conv_58_dequantized, var_568_dequantized_transposed,
        {
            bias: var_569_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 384, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_567_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.012513305])
    );

    const var_567_quantized = builder.quantizeLinear(
        var_567_QuantizeInput,
        var_567_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_567_Conv_60_dequantized = builder.dequantizeLinear(
        var_567_quantized,
        var_567_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_571_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.00487904])
    );

    const var_571_dequantized = builder.dequantizeLinear(
        var_571_quantized,
        var_571_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_572_dequantized = builder.dequantizeLinear(
        var_572_quantized,
        var_572_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_571_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [96, 1, 1, 384]},
        new Int8Array(weights_array_buffer, 2150604, 36864 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_571_dequantized_transposed = builder.dequantizeLinear(
        var_571_quantized_transposed,
        var_571_scale,
        var_475_zero_point
    );

    const var_570_QuantizeInput = builder.conv2d(
        var_567_Conv_60_dequantized, var_571_dequantized_transposed,
        {
            bias: var_572_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_570_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.024123587])
    );

    const var_624_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([123])
    );

    const var_570_quantized = builder.quantizeLinear(
        var_570_QuantizeInput,
        var_570_scale_reshaped_1_1_1_1,
        var_624_zero_point_reshaped_1_1_1_1
    );

    const var_570_duplicated = builder.dequantizeLinear(
        var_570_quantized,
        var_570_scale_reshaped_1_1_1_1,
        var_624_zero_point_reshaped_1_1_1_1
    );

    const var_574_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0013143878])
    );

    const var_574_dequantized = builder.dequantizeLinear(
        var_574_quantized,
        var_574_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_575_dequantized = builder.dequantizeLinear(
        var_575_quantized,
        var_575_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_574_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [576, 1, 1, 96]},
        new Int8Array(weights_array_buffer, 2187856, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_574_dequantized_transposed = builder.dequantizeLinear(
        var_574_quantized_transposed,
        var_574_scale,
        var_475_zero_point
    );

    const var_573_QuantizeInput = builder.conv2d(
        var_570_duplicated, var_574_dequantized_transposed,
        {
            bias: var_575_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_573_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0067103086])
    );

    const var_573_quantized = builder.quantizeLinear(
        var_573_QuantizeInput,
        var_573_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_573_Conv_63_dequantized = builder.dequantizeLinear(
        var_573_quantized,
        var_573_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_577_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.07777221])
    );

    const var_577_dequantized = builder.dequantizeLinear(
        var_577_quantized,
        var_577_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_578_dequantized = builder.dequantizeLinear(
        var_578_quantized,
        var_578_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_577_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 576]},
        new Int8Array(weights_array_buffer, 2245460, 5184 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_577_dequantized_transposed = builder.dequantizeLinear(
        var_577_quantized_transposed,
        var_577_scale,
        var_475_zero_point
    );

    const var_576_QuantizeInput = builder.conv2d(
        var_573_Conv_63_dequantized, var_577_dequantized_transposed,
        {
            bias: var_578_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 576, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_576_quantized = builder.quantizeLinear(
        var_576_QuantizeInput,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_576_Conv_65_dequantized = builder.dequantizeLinear(
        var_576_quantized,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_580_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.003954809])
    );

    const var_580_dequantized = builder.dequantizeLinear(
        var_580_quantized,
        var_580_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_581_dequantized = builder.dequantizeLinear(
        var_581_quantized,
        var_581_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_580_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [96, 1, 1, 576]},
        new Int8Array(weights_array_buffer, 2252952, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_580_dequantized_transposed = builder.dequantizeLinear(
        var_580_quantized_transposed,
        var_580_scale,
        var_475_zero_point
    );

    const var_579_QuantizeInput = builder.conv2d(
        var_576_Conv_65_dequantized, var_580_dequantized_transposed,
        {
            bias: var_581_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_579_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.02798538])
    );

    const var_579_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([115])
    );

    const var_579_quantized = builder.quantizeLinear(
        var_579_QuantizeInput,
        var_579_scale_reshaped_1_1_1_1,
        var_579_zero_point_reshaped_1_1_1_1
    );

    const var_579 = builder.dequantizeLinear(
        var_579_quantized,
        var_579_scale_reshaped_1_1_1_1,
        var_579_zero_point_reshaped_1_1_1_1
    );

    const var_417 = builder.add(var_570_duplicated, var_579);

    const var_417_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.031824797])
    );

    const var_417_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([109])
    );

    const var_417_Conv_67_QuantizeLinear = builder.quantizeLinear(
        var_417,
        var_417_scale_reshaped_1_1_1_1,
        var_417_zero_point_reshaped_1_1_1_1
    );

    const var_417_Conv_67_dequantized = builder.dequantizeLinear(
        var_417_Conv_67_QuantizeLinear,
        var_417_scale_reshaped_1_1_1_1,
        var_417_zero_point_reshaped_1_1_1_1
    );

    const var_583_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0017125434])
    );

    const var_583_dequantized = builder.dequantizeLinear(
        var_583_quantized,
        var_583_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_584_dequantized = builder.dequantizeLinear(
        var_584_quantized,
        var_584_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_583_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [576, 1, 1, 96]},
        new Int8Array(weights_array_buffer, 2462236, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_583_dequantized_transposed = builder.dequantizeLinear(
        var_583_quantized_transposed,
        var_583_scale,
        var_475_zero_point
    );

    const var_582_QuantizeInput = builder.conv2d(
        var_417_Conv_67_dequantized, var_583_dequantized_transposed,
        {
            bias: var_584_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_582_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.018577984])
    );

    const var_582_quantized = builder.quantizeLinear(
        var_582_QuantizeInput,
        var_582_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_582_Conv_69_dequantized = builder.dequantizeLinear(
        var_582_quantized,
        var_582_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_586_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.06750055])
    );

    const var_586_dequantized = builder.dequantizeLinear(
        var_586_quantized,
        var_586_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_587_dequantized = builder.dequantizeLinear(
        var_587_quantized,
        var_587_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_586_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 576]},
        new Int8Array(weights_array_buffer, 2519840, 5184 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_586_dequantized_transposed = builder.dequantizeLinear(
        var_586_quantized_transposed,
        var_586_scale,
        var_475_zero_point
    );

    const var_585_QuantizeInput = builder.conv2d(
        var_582_Conv_69_dequantized, var_586_dequantized_transposed,
        {
            bias: var_587_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 576, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_585_quantized = builder.quantizeLinear(
        var_585_QuantizeInput,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_585_Conv_71_dequantized = builder.dequantizeLinear(
        var_585_quantized,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_589_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.009840593])
    );

    const var_589_dequantized = builder.dequantizeLinear(
        var_589_quantized,
        var_589_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_590_dequantized = builder.dequantizeLinear(
        var_590_quantized,
        var_590_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_589_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [96, 1, 1, 576]},
        new Int8Array(weights_array_buffer, 2527332, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_589_dequantized_transposed = builder.dequantizeLinear(
        var_589_quantized_transposed,
        var_589_scale,
        var_475_zero_point
    );

    const var_588_QuantizeInput = builder.conv2d(
        var_585_Conv_71_dequantized, var_589_dequantized_transposed,
        {
            bias: var_590_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_588_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.06331735])
    );

    const var_588_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([142])
    );

    const var_588_quantized = builder.quantizeLinear(
        var_588_QuantizeInput,
        var_588_scale_reshaped_1_1_1_1,
        var_588_zero_point_reshaped_1_1_1_1
    );

    const var_588 = builder.dequantizeLinear(
        var_588_quantized,
        var_588_scale_reshaped_1_1_1_1,
        var_588_zero_point_reshaped_1_1_1_1
    );

    const var_426 = builder.add(var_417, var_588);

    const var_426_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.07048993])
    );

    const var_426_Conv_73_QuantizeLinear = builder.quantizeLinear(
        var_426,
        var_426_scale_reshaped_1_1_1_1,
        var_624_zero_point_reshaped_1_1_1_1
    );

    const var_426_Conv_73_dequantized = builder.dequantizeLinear(
        var_426_Conv_73_QuantizeLinear,
        var_426_scale_reshaped_1_1_1_1,
        var_624_zero_point_reshaped_1_1_1_1
    );

    const var_592_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0014183413])
    );

    const var_592_dequantized = builder.dequantizeLinear(
        var_592_quantized,
        var_592_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_593_dequantized = builder.dequantizeLinear(
        var_593_quantized,
        var_593_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_592_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [576, 1, 1, 96]},
        new Int8Array(weights_array_buffer, 2583020, 55296 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_592_dequantized_transposed = builder.dequantizeLinear(
        var_592_quantized_transposed,
        var_592_scale,
        var_475_zero_point
    );

    const var_591_QuantizeInput = builder.conv2d(
        var_426_Conv_73_dequantized, var_592_dequantized_transposed,
        {
            bias: var_593_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_591_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.014397864])
    );

    const var_591_quantized = builder.quantizeLinear(
        var_591_QuantizeInput,
        var_591_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_591_Conv_75_dequantized = builder.dequantizeLinear(
        var_591_quantized,
        var_591_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_595_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.018182425])
    );

    const var_595_dequantized = builder.dequantizeLinear(
        var_595_quantized,
        var_595_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_596_dequantized = builder.dequantizeLinear(
        var_596_quantized,
        var_596_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_595_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 576]},
        new Int8Array(weights_array_buffer, 2644464, 5184 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_595_dequantized_transposed = builder.dequantizeLinear(
        var_595_quantized_transposed,
        var_595_scale,
        var_475_zero_point
    );

    const var_594_QuantizeInput = builder.conv2d(
        var_591_Conv_75_dequantized, var_595_dequantized_transposed,
        {
            bias: var_596_dequantized, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 576, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_594_quantized = builder.quantizeLinear(
        var_594_QuantizeInput,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_594_Conv_77_dequantized = builder.dequantizeLinear(
        var_594_quantized,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_598_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0025079844])
    );

    const var_598_dequantized = builder.dequantizeLinear(
        var_598_quantized,
        var_598_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_599_dequantized = builder.dequantizeLinear(
        var_599_quantized,
        var_599_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_598_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [160, 1, 1, 576]},
        new Int8Array(weights_array_buffer, 2967796, 92160 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_598_dequantized_transposed = builder.dequantizeLinear(
        var_598_quantized_transposed,
        var_598_scale,
        var_475_zero_point
    );

    const var_597_QuantizeInput = builder.conv2d(
        var_594_Conv_77_dequantized, var_598_dequantized_transposed,
        {
            bias: var_599_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_597_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.06929498])
    );

    const var_597_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([149])
    );

    const var_597_quantized = builder.quantizeLinear(
        var_597_QuantizeInput,
        var_597_scale_reshaped_1_1_1_1,
        var_597_zero_point_reshaped_1_1_1_1
    );

    const var_597_duplicated = builder.dequantizeLinear(
        var_597_quantized,
        var_597_scale_reshaped_1_1_1_1,
        var_597_zero_point_reshaped_1_1_1_1
    );

    const var_601_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0029125263])
    );

    const var_601_dequantized = builder.dequantizeLinear(
        var_601_quantized,
        var_601_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_602_dequantized = builder.dequantizeLinear(
        var_602_quantized,
        var_602_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_601_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [960, 1, 1, 160]},
        new Int8Array(weights_array_buffer, 3060600, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_601_dequantized_transposed = builder.dequantizeLinear(
        var_601_quantized_transposed,
        var_601_scale,
        var_475_zero_point
    );

    const var_600_QuantizeInput = builder.conv2d(
        var_597_duplicated, var_601_dequantized_transposed,
        {
            bias: var_602_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_600_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.020971533])
    );

    const var_600_quantized = builder.quantizeLinear(
        var_600_QuantizeInput,
        var_600_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_600_Conv_80_dequantized = builder.dequantizeLinear(
        var_600_quantized,
        var_600_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_604_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.07850213])
    );

    const var_604_dequantized = builder.dequantizeLinear(
        var_604_quantized,
        var_604_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_605_dequantized = builder.dequantizeLinear(
        var_605_quantized,
        var_605_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_604_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 960]},
        new Int8Array(weights_array_buffer, 3218044, 8640 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_604_dequantized_transposed = builder.dequantizeLinear(
        var_604_quantized_transposed,
        var_604_scale,
        var_475_zero_point
    );

    const var_603_QuantizeInput = builder.conv2d(
        var_600_Conv_80_dequantized, var_604_dequantized_transposed,
        {
            bias: var_605_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 960, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_603_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.022908082])
    );

    const var_603_quantized = builder.quantizeLinear(
        var_603_QuantizeInput,
        var_603_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_603_Conv_82_dequantized = builder.dequantizeLinear(
        var_603_quantized,
        var_603_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_607_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.002574469])
    );

    const var_607_dequantized = builder.dequantizeLinear(
        var_607_quantized,
        var_607_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_608_dequantized = builder.dequantizeLinear(
        var_608_quantized,
        var_608_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_607_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [160, 1, 1, 960]},
        new Int8Array(weights_array_buffer, 3230528, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_607_dequantized_transposed = builder.dequantizeLinear(
        var_607_quantized_transposed,
        var_607_scale,
        var_475_zero_point
    );

    const var_606_QuantizeInput = builder.conv2d(
        var_603_Conv_82_dequantized, var_607_dequantized_transposed,
        {
            bias: var_608_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_606_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.06415011])
    );

    const var_606_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([105])
    );

    const var_606_quantized = builder.quantizeLinear(
        var_606_QuantizeInput,
        var_606_scale_reshaped_1_1_1_1,
        var_606_zero_point_reshaped_1_1_1_1
    );

    const var_606 = builder.dequantizeLinear(
        var_606_quantized,
        var_606_scale_reshaped_1_1_1_1,
        var_606_zero_point_reshaped_1_1_1_1
    );

    const var_443 = builder.add(var_597_duplicated, var_606);

    const var_443_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.12168143])
    );

    const var_443_Conv_84_QuantizeLinear = builder.quantizeLinear(
        var_443,
        var_443_scale_reshaped_1_1_1_1,
        var_480_zero_point_reshaped_1_1_1_1
    );

    const var_443_Conv_84_dequantized = builder.dequantizeLinear(
        var_443_Conv_84_QuantizeLinear,
        var_443_scale_reshaped_1_1_1_1,
        var_480_zero_point_reshaped_1_1_1_1
    );

    const var_610_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0016836411])
    );

    const var_610_dequantized = builder.dequantizeLinear(
        var_610_quantized,
        var_610_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_611_dequantized = builder.dequantizeLinear(
        var_611_quantized,
        var_611_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_610_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [960, 1, 1, 160]},
        new Int8Array(weights_array_buffer, 3384772, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_610_dequantized_transposed = builder.dequantizeLinear(
        var_610_quantized_transposed,
        var_610_scale,
        var_475_zero_point
    );

    const var_609_QuantizeInput = builder.conv2d(
        var_443_Conv_84_dequantized, var_610_dequantized_transposed,
        {
            bias: var_611_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_609_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.02009347])
    );

    const var_609_quantized = builder.quantizeLinear(
        var_609_QuantizeInput,
        var_609_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_609_Conv_86_dequantized = builder.dequantizeLinear(
        var_609_quantized,
        var_609_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_613_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.057705373])
    );

    const var_613_dequantized = builder.dequantizeLinear(
        var_613_quantized,
        var_613_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_614_dequantized = builder.dequantizeLinear(
        var_614_quantized,
        var_614_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_613_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 960]},
        new Int8Array(weights_array_buffer, 2649648, 8640 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_613_dequantized_transposed = builder.dequantizeLinear(
        var_613_quantized_transposed,
        var_613_scale,
        var_475_zero_point
    );

    const var_612_QuantizeInput = builder.conv2d(
        var_609_Conv_86_dequantized, var_613_dequantized_transposed,
        {
            bias: var_614_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 960, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_612_quantized = builder.quantizeLinear(
        var_612_QuantizeInput,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_612_Conv_88_dequantized = builder.dequantizeLinear(
        var_612_quantized,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_616_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.004802136])
    );

    const var_616_dequantized = builder.dequantizeLinear(
        var_616_quantized,
        var_616_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_617_dequantized = builder.dequantizeLinear(
        var_617_quantized,
        var_617_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_616_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [160, 1, 1, 960]},
        new Int8Array(weights_array_buffer, 2308636, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_616_dequantized_transposed = builder.dequantizeLinear(
        var_616_quantized_transposed,
        var_616_scale,
        var_475_zero_point
    );

    const var_615_QuantizeInput = builder.conv2d(
        var_612_Conv_88_dequantized, var_616_dequantized_transposed,
        {
            bias: var_617_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_615_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.1445715])
    );

    const var_615_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([131])
    );

    const var_615_quantized = builder.quantizeLinear(
        var_615_QuantizeInput,
        var_615_scale_reshaped_1_1_1_1,
        var_615_zero_point_reshaped_1_1_1_1
    );

    const var_615 = builder.dequantizeLinear(
        var_615_quantized,
        var_615_scale_reshaped_1_1_1_1,
        var_615_zero_point_reshaped_1_1_1_1
    );

    const var_452 = builder.add(var_443, var_615);

    const var_452_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.23722179])
    );

    const var_452_zero_point_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1, 1, 1]},
        new Uint8Array([110])
    );

    const var_452_Conv_90_QuantizeLinear = builder.quantizeLinear(
        var_452,
        var_452_scale_reshaped_1_1_1_1,
        var_452_zero_point_reshaped_1_1_1_1
    );

    const var_452_Conv_90_dequantized = builder.dequantizeLinear(
        var_452_Conv_90_QuantizeLinear,
        var_452_scale_reshaped_1_1_1_1,
        var_452_zero_point_reshaped_1_1_1_1
    );

    const var_619_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.0010381987])
    );

    const var_619_dequantized = builder.dequantizeLinear(
        var_619_quantized,
        var_619_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_620_dequantized = builder.dequantizeLinear(
        var_620_quantized,
        var_620_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_619_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [960, 1, 1, 160]},
        new Int8Array(weights_array_buffer, 1875824, 153600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_619_dequantized_transposed = builder.dequantizeLinear(
        var_619_quantized_transposed,
        var_619_scale,
        var_475_zero_point
    );

    const var_618_QuantizeInput = builder.conv2d(
        var_452_Conv_90_dequantized, var_619_dequantized_transposed,
        {
            bias: var_620_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_618_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.015018467])
    );

    const var_618_quantized = builder.quantizeLinear(
        var_618_QuantizeInput,
        var_618_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_618_Conv_92_dequantized = builder.dequantizeLinear(
        var_618_quantized,
        var_618_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_622_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.07043054])
    );

    const var_622_dequantized = builder.dequantizeLinear(
        var_622_quantized,
        var_622_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_623_dequantized = builder.dequantizeLinear(
        var_623_quantized,
        var_623_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_622_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1, 3, 3, 960]},
        new Int8Array(weights_array_buffer, 2085372, 8640 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_622_dequantized_transposed = builder.dequantizeLinear(
        var_622_quantized_transposed,
        var_622_scale,
        var_475_zero_point
    );

    const var_621_QuantizeInput = builder.conv2d(
        var_618_Conv_92_dequantized, var_622_dequantized_transposed,
        {
            bias: var_623_dequantized, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 960, filterLayout: 'ihwo', inputLayout: 'nhwc'
        }
    );

    const var_621_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.008207709])
    );

    const var_621_quantized = builder.quantizeLinear(
        var_621_QuantizeInput,
        var_621_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_621_Conv_94_dequantized = builder.dequantizeLinear(
        var_621_quantized,
        var_621_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_625_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.00507597])
    );

    const var_625_dequantized = builder.dequantizeLinear(
        var_625_quantized,
        var_625_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_626_dequantized = builder.dequantizeLinear(
        var_626_quantized,
        var_626_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_625_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [320, 1, 1, 960]},
        new Int8Array(weights_array_buffer, 2660596, 307200 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_625_dequantized_transposed = builder.dequantizeLinear(
        var_625_quantized_transposed,
        var_625_scale,
        var_475_zero_point
    );

    const var_624_QuantizeInput = builder.conv2d(
        var_621_Conv_94_dequantized, var_625_dequantized_transposed,
        {
            bias: var_626_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_624_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.014525251])
    );

    const var_624_quantized = builder.quantizeLinear(
        var_624_QuantizeInput,
        var_624_scale_reshaped_1_1_1_1,
        var_624_zero_point_reshaped_1_1_1_1
    );

    const var_624_Conv_95_dequantized = builder.dequantizeLinear(
        var_624_quantized,
        var_624_scale_reshaped_1_1_1_1,
        var_624_zero_point_reshaped_1_1_1_1
    );

    const var_628_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.013814686])
    );

    const var_628_dequantized = builder.dequantizeLinear(
        var_628_quantized,
        var_628_scale_reshaped_1_1_1_1,
        var_475_zero_point_reshaped_1_1_1_1
    );

    const var_629_dequantized = builder.dequantizeLinear(
        var_629_quantized,
        var_629_scale,
        var_623_zero_point
    );

    // Re-create constant operand from transposed weights.

    const var_628_quantized_transposed = builder.constant(
        {dataType: 'int8', shape: [1280, 1, 1, 320]},
        new Int8Array(weights_array_buffer, 1302632, 409600 / Int8Array.BYTES_PER_ELEMENT)
    );

    const var_628_dequantized_transposed = builder.dequantizeLinear(
        var_628_quantized_transposed,
        var_628_scale,
        var_475_zero_point
    );

    const var_627_QuantizeInput = builder.conv2d(
        var_624_Conv_95_dequantized, var_628_dequantized_transposed,
        {
            bias: var_629_dequantized, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, filterLayout: 'ohwi', inputLayout: 'nhwc'
        }
    );

    const var_627_quantized = builder.quantizeLinear(
        var_627_QuantizeInput,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_627_duplicated = builder.dequantizeLinear(
        var_627_quantized,
        var_576_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_464_QuantizeInput = builder.averagePool2d(
        var_627_duplicated, { layout: 'nhwc' }
    );

    const var_464_scale_reshaped_1_1_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1, 1, 1]},
        new Float32Array([0.019110054])
    );

    const var_464_quantized = builder.quantizeLinear(
        var_464_QuantizeInput,
        var_464_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_464_Reshape_103_dequantized = builder.dequantizeLinear(
        var_464_quantized,
        var_464_scale_reshaped_1_1_1_1,
        var_474_zero_point_reshaped_1_1_1_1
    );

    const var_472_QuantizeInput = builder.reshape(
        var_464_Reshape_103_dequantized,
        (() => {
        const shape = [1, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var_464_Reshape_103_dequantized.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var_464_scale_reshaped_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1]},
        new Float32Array([0.019110054])
    );

    const var_474_zero_point_reshaped_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1]},
        new Uint8Array([0])
    );

    const var_472_quantized = builder.quantizeLinear(
        var_472_QuantizeInput,
        var_464_scale_reshaped_1_1,
        var_474_zero_point_reshaped_1_1
    );

    const var_472_Gemm_104_MatMul_dequantized = builder.dequantizeLinear(
        var_472_quantized,
        var_464_scale_reshaped_1_1,
        var_474_zero_point_reshaped_1_1
    );

    const var_classifier_1_weight_scale_reshaped_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1]},
        new Float32Array([0.0026049719])
    );

    const var_475_zero_point_reshaped_1_1 = builder.constant(
        {dataType: 'int8', shape: [1, 1]},
        new Int8Array([0])
    );

    const var_classifier_1_weight_dequantized = builder.dequantizeLinear(
        var_classifier_1_weight_quantized,
        var_classifier_1_weight_scale_reshaped_1_1,
        var_475_zero_point_reshaped_1_1
    );

    const output_MatMul_QuantizeInput = builder.matmul(var_472_Gemm_104_MatMul_dequantized, var_classifier_1_weight_dequantized);

    const output_MatMul_scale_reshaped_1_1 = builder.constant(
        {dataType: 'float32', shape: [1, 1]},
        new Float32Array([0.16079935])
    );

    const output_MatMul_zero_point_reshaped_1_1 = builder.constant(
        {dataType: 'uint8', shape: [1, 1]},
        new Uint8Array([76])
    );

    const output_MatMul_quantized = builder.quantizeLinear(
        output_MatMul_QuantizeInput,
        output_MatMul_scale_reshaped_1_1,
        output_MatMul_zero_point_reshaped_1_1
    );

    const output_MatMul = builder.dequantizeLinear(
        output_MatMul_quantized,
        output_MatMul_scale_reshaped_1_1,
        output_MatMul_zero_point_reshaped_1_1
    );

    const output = builder.softmax(builder.add(output_MatMul, var_classifier_1_bias));

    // Build graph with output operands.

    this.graph_ = await builder.build({'output': output});

    // Create graph output tensors.

    this.outputTensors_['output'] = await this.context_.createTensor(
        {dataType: output.dataType, shape: output.shape, readable: true}
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

        case 'float16': typedArrayCtor = Float16Array; break;

        case 'int32': typedArrayCtor = Int32Array; break;

        case 'uint8': typedArrayCtor = Uint8Array; break;

        case 'int8': typedArrayCtor = Int8Array; break;

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